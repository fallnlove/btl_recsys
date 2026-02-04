from src.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm

from torch.utils.data import DataLoader
from src.metrics import NDCGMetric

def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_seqs(user_items, maxlen, pad_token):
    seq = np.full(maxlen, pad_token, dtype=np.int32)
    pos = np.full(maxlen, pad_token, dtype=np.int32)
    neg = np.empty((maxlen, 1))

    if len(user_items) <= maxlen:
        if len(user_items) > 1:
            seq[-len(user_items) + 1:] = user_items[:-1]
        if len(user_items) > 0:
            pos[-len(user_items):] = user_items
    else:
        seq = user_items[-maxlen-1:-1]
        pos = user_items[-maxlen:]

    return seq, pos, neg
    
def collate_fn_sasrec(x, maxlen, pad_token, is_train):

    seqs_batch = []
    pos_batch = []
    user_ids = []
    seen_history = []

    for user in x:
        seq, pos, neg = pad_seqs(user["history"].tolist(), maxlen, pad_token)

        seqs_batch.append(seq)
        pos_batch.append(pos)
        user_ids.append(user["user_id"])
        if not is_train:
            seen_history.append(user["history"])

    batch = {
        "seq": torch.LongTensor(np.asarray(seqs_batch)),
        "pos": torch.LongTensor(np.asarray(pos_batch)),
        "user_id": torch.LongTensor(np.asarray(user_ids)),
        "seen_history": seen_history
    }

    return batch


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.ff = nn.Sequential(
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            torch.nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        outputs = self.ff(
            inputs.transpose(-1,-2)
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRecBackBone(nn.Module):
    def __init__(
            self,
            item_num,
            hidden_units,
            dropout_rate,
            maxlen,
            num_blocks,
            num_heads,
            manual_seed=37
        ):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, hidden_units, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                hidden_units,num_heads,dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(manual_seed)
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token

        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :, :] * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def score(self, seq):

        log_feats = self.log2feats(seq)
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRecModel(BaseModel):
    def __init__(
        self,
        name: str = "sasrec",
        hidden_size: int = 64,
        num_blocks: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        lr: float = 5e-3,
        device: str = "cuda",
        seq_len: int = 50,
        n_epochs: int = 2000,
        batch_size: int = 64,
        seed: int = 52,
        log_step: int = 200,
        mix_x: bool = True,
        alpha: float = 2.0,
        beta: float = 1.0,
        bucket_size_y: int = 256,
        patience_per_epoch: int = 1,
        max_patience: int = 50,
        val_top_n: int = 10,
        filter_seen: bool = True
    ):
        BaseModel.__init__(self, name)

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.device = device
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.filter_seen = filter_seen

        self.alpha = alpha
        self.beta = beta

        self.patience_per_epoch = patience_per_epoch
        self.max_patience = max_patience

        self.mix_x = mix_x

        self.bucket_size_y = bucket_size_y

        self.log_step = log_step
        self.val_top_n = val_top_n
    
    def _post_init(self, train_dataset, val_dataset):
        self.n_items = train_dataset.n_items

        self.bucket_size_x = int(2 * (self.batch_size * self.seq_len) ** 0.5)
        self.n_buckets = int(2 * (self.batch_size * self.seq_len) ** 0.5)

        print("Calculated bucket size:", self.bucket_size_x)
        print("Calculated n buckets:", self.n_buckets)

        self.model = SASRecBackBone(
            self.n_items,
            self.hidden_size,
            self.dropout,
            self.seq_len,
            self.num_blocks,
            self.num_heads,
            manual_seed=37
        )

        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def _sasrec_forward(self, log_seqs, pos_seqs):

        emb = self.model.log2feats(log_seqs)


        hd = emb.shape[-1]

        x = emb.view(-1, hd)
        y = pos_seqs.view(-1)
        w = self.model.item_emb.weight

        correct_class_logits_ = (x * torch.index_select(w, dim=0, index=y)).sum(dim=1) # (bs,)

        with torch.no_grad():
            if self.mix_x:
                omega = 1/np.sqrt(np.sqrt(hd)) * torch.randn(x.shape[0], self.n_buckets, device=x.device)
                buckets = omega.T @ x
                del omega
            else:
                buckets = 1/np.sqrt(np.sqrt(hd)) * torch.randn(self.n_buckets, hd, device=x.device) # (n_b, hd)

        with torch.no_grad():
            x_bucket = buckets @ x.T # (n_b, hd) x (hd, b) -> (n_b, b)
            x_bucket[:, log_seqs.view(-1) == self.model.pad_token] = float('-inf')
            _, top_x_bucket = torch.topk(x_bucket, dim=1, k=self.bucket_size_x) # (n_b, bs_x)
            del x_bucket

            y_bucket = buckets @ w.T # (n_b, hd) x (hd, n_cl) -> (n_b, n_cl)

            y_bucket[:, self.model.pad_token] = float('-inf')
            _, top_y_bucket = torch.topk(y_bucket, dim=1, k=self.bucket_size_y) # (n_b, bs_y)
            del y_bucket

        x_bucket = torch.gather(x, 0, top_x_bucket.view(-1, 1).expand(-1, hd)).view(self.n_buckets, self.bucket_size_x, hd) # (n_b, bs_x, hd)
        y_bucket = torch.gather(w, 0, top_y_bucket.view(-1, 1).expand(-1, hd)).view(self.n_buckets, self.bucket_size_y, hd) # (n_b, bs_y, hd)
        
        wrong_class_logits = (x_bucket @ y_bucket.transpose(-1, -2)) # (n_b, bs_x, bs_y)
        mask = torch.index_select(y, dim=0, index=top_x_bucket.view(-1)).view(self.n_buckets, self.bucket_size_x)[:, :, None] == top_y_bucket[:, None, :] # (n_b, bs_x, bs_y)
        wrong_class_logits = wrong_class_logits.masked_fill(mask, float('-inf')) # (n_b, bs_x, bs_y)
        correct_class_logits = torch.index_select(correct_class_logits_, dim=0, index=top_x_bucket.view(-1)).view(self.n_buckets, self.bucket_size_x)[:, :, None] # (n_b, bs_x, 1)
        logits = torch.cat((wrong_class_logits, correct_class_logits), dim=2) # (n_b, bs_x, bs_y + 1)

        loss_ = F.cross_entropy(logits.view(-1, logits.shape[-1]), (logits.shape[-1] - 1) * torch.ones(logits.shape[0] * logits.shape[1], dtype=torch.int64, device=logits.device), reduction='none') # (n_b * bs_x,)
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        loss.scatter_reduce_(0, top_x_bucket.view(-1), loss_, reduce='amax', include_self=False)

        loss = loss[(loss != 0) & (y != self.model.pad_token)]
        loss = torch.mean(loss)

        return loss
    
    def suggest_additional_params(self):
        return {"n_epochs": self.n_epochs}
            
    
    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """

        self._post_init(
            train_dataset,
            val_dataset
        )

        train_dataloader = DataLoader(
            train_dataset,
            self.batch_size,
            shuffle = True,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, True)
        )

        best_metric = -1
        patience_cnt = 0
        best_epoch = -1
        best_state_dict = None

        for i in trange(self.n_epochs):
            self.model.train()

            train_loss = []

            index = 0

            for batch in tqdm(train_dataloader):

                loss = self._sasrec_forward(
                    batch["seq"].to(self.device),
                    batch["pos"].to(self.device),
                )

                self.opt.zero_grad()

                train_loss.append(loss.item())

                loss.backward()
                
                self.opt.step() 

                index += 1

                if (index + 1) % self.log_step == 0:
                    print(f"Mean train loss: {sum(train_loss[-self.log_step:]) / self.log_step}")    
            
            
            print(f"Mean train loss: {sum(train_loss) / len(train_loss)}")

            if (i + 1) % self.patience_per_epoch == 0 and val_dataset is not None:

                holdout_users = val_dataset.get_holdout_users()
        
                predictions = self.predict(
                    val_dataset,
                    self.val_top_n
                )

                

                metric = NDCGMetric(self.val_top_n)
                metric_val = metric(
                    predictions[holdout_users, :],
                    val_dataset.get_holdout_array()[holdout_users],
                )

                print(f"Value of {metric.name}: {metric_val}")

                if best_metric < metric_val:
                    best_metric = metric_val
                    patience_cnt = 0

                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }

                    best_epoch = i + 1
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.max_patience:
                        break

        if best_metric >= 0:
            self.model.load_state_dict(best_state_dict)
            self.n_epochs = best_epoch





                
    @torch.no_grad()
    def predict(self, dataset, top_n: int, batch_size: int = 128):
        self.model.eval()

        n_users = dataset.n_users
        
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle = False,
            collate_fn = lambda x: collate_fn_sasrec(x, self.seq_len, self.n_items, False)
        )
    
        recommendations = np.zeros((n_users, top_n))

        for batch in dataloader:

            logits = self.model.score(
                seq=batch["pos"].to(self.device),
            )

            if self.filter_seen:
                for i in range(len(batch["seen_history"])):
                    if batch["seen_history"][i].shape[0] > 0:
                        logits[i][batch["seen_history"][i]] = -10 ** 9
    
            top_items = torch.topk(
                logits,
                k=top_n,
                dim=1
            ).indices
    
            for uid, recs in zip(batch["user_id"], top_items):
                recommendations[uid, :] = recs.cpu().numpy()
                
        return recommendations
    
