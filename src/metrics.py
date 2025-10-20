import inspect

import torch


class MetaParent(type):

    def __init__(cls, name, base, params, **kwargs):
        super().__init__(name, base, params)
        is_base_class = cls.mro()[1] is object
        if is_base_class:
            base_class = cls
        else:
            base_class_found = False
            for key in cls.mro():
                if isinstance(key, MetaParent) and key.mro()[1] is object:
                    assert base_class_found is False, "multiple base classes(bug)"
                    base_class = key
                    base_class_found = True
            assert base_class_found is True, f"no base class for {name}"

        if is_base_class:
            cls._subclasses = {}

        @classmethod
        def __init_subclass__(scls, config_name=None):
            super().__init_subclass__()
            if config_name is not None:
                if config_name in base_class._subclasses:
                    raise ValueError(
                        "Class with name `{}` is already registered".format(config_name)
                    )
                scls.config_name = config_name
                base_class._subclasses[config_name] = scls

        cls.__init_subclass__ = __init_subclass__

        @classmethod
        def parent_create_from_config(cls, config, **kwargs):
            if "type" in config:
                return cls._subclasses[config["type"]].create_from_config(
                    config, **kwargs
                )
            else:
                raise ValueError(
                    "There is no `type` provided for the `{}` class".format(name)
                )

        # Take kwargs for the last initialized baseclass
        init_kwargs = {}
        for bcls in cls.mro()[:-1]:  # Look into all base classes except object
            if "__init__" not in bcls.__dict__:
                continue
            init_kwargs = inspect.signature(bcls.__init__).parameters
            break

        @classmethod
        def child_create_from_config(cls, config, **kwargs):
            kwargs = {}
            for key, argspec in init_kwargs.items():
                if key == "self":
                    continue
                value = config.get(key, argspec.default)
                if value is inspect.Parameter.empty:
                    msg = "There is no value for `{}.__init__` required field `{}` in config `{}`"
                    raise ValueError(msg.format(cls, key, config))
                kwargs[key] = value
            return cls(**kwargs)

        if "create_from_config" not in cls.__dict__:
            cls.create_from_config = (
                parent_create_from_config if is_base_class else child_create_from_config
            )


class BaseMetric(metaclass=MetaParent):
    pass


class StatefullMetric(BaseMetric):

    def reduce(self):
        raise NotImplementedError


class StaticMetric(BaseMetric, config_name="dummy"):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs


class CompositeMetric(BaseMetric, config_name="composite"):

    def __init__(self, metrics):
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        return cls(
            metrics=[BaseMetric.create_from_config(cfg) for cfg in config["metrics"]]
        )

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric(BaseMetric, config_name="ndcg"):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs["logits"][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        labels = inputs["labels.ids"].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions, labels[..., None]
        ).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(
            torch.arange(1, self._k + 1, 1).float() + 1.0
        ).to(
            hits.device
        )  # (k)
        dcg = hits @ discount_factor  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric(BaseMetric, config_name="recall"):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs):
        predictions = inputs["logits"][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        labels = inputs["labels.ids"].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(
            predictions, labels[..., None]
        ).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class CoverageMetric(StatefullMetric, config_name="coverage"):

    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(k=config["k"], num_items=kwargs["num_items"])

    def __call__(self, inputs):
        predictions = inputs["logits"][
            :, : self._k
        ].float()  # (batch_size, top_k_indices)
        return predictions.view(-1).long().cpu().detach().tolist()  # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items
