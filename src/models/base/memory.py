# memory.py

import torch
from torch import nn
import torch.nn.functional as F

from src.models.core.nn_utils import cosine_sim


class SketchMem(nn.Module):
    """
    Overview:
        Input query -> "factorize" into base and category-specific queries
    """
    def __init__(self, base_mem_size=100, category_mem_size=5,
                 mem_dim=64, input_dim=None, output_dim=None):
        super().__init__()
        self.base_mem_size = base_mem_size
        self.category_mem_size = base_mem_size
        self.mem_dim = mem_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Model params
        self.input_base_fc = nn.Linear(input_dim, mem_dim)
        self.input_category_fc = nn.Linear(input_dim, mem_dim)

        self.base_mem_keys = nn.Parameter(torch.FloatTensor(base_mem_size, mem_dim))
        self.base_mem_vals = nn.Parameter(torch.FloatTensor(base_mem_size, mem_dim))
        self.category_mem_keys = nn.Parameter(torch.FloatTensor(35, category_mem_size, mem_dim))  # 35 final categories
        self.category_mem_vals = nn.Parameter(torch.FloatTensor(35, category_mem_size, mem_dim))

        self.output_fc = nn.Linear(mem_dim * 2, output_dim)

        # Init
        nn.init.kaiming_uniform_(self.base_mem_keys)
        nn.init.kaiming_uniform_(self.base_mem_vals)
        nn.init.kaiming_uniform_(self.category_mem_keys)
        nn.init.kaiming_uniform_(self.category_mem_vals)

    def forward(self, query, category_idxs):
        """
        Args:
            query ([bsz, input_dim]]
            category_idxs ([bsz]): index of which category for each batch item

        Returns:
            result ([bsz, output_dim])
        """
        # Look up base memories
        base_query = self.input_base_fc(query)  # [bsz, mem_dim]
        base_sims = cosine_sim(base_query, self.base_mem_keys)  # [bsz, base_mem_size]
        base_sims = F.softmax(base_sims, dim=-1)
        base_lookup = base_sims.mm(self.base_mem_vals)  # [bsz, mem_dim]

        # Look up category-specific memories.
        # First index into category for each item in batch
        category_query = self.input_category_fc(query)  # [bsz, mem_dim]
        category_mem_keys = self.category_mem_keys.index_select(0, category_idxs)  # [bsz, category_mem_size, mem_dim]
        category_query = category_query.unsqueeze(1)  # [bsz, 1, mem_dim]
        category_sims = category_query.bmm(category_mem_keys.transpose(1,2))  # [bsz, 1, category_mem_size]  # Bmm([b * n * m], [b * m * p]) -> [b * n * p]
        # category_sims = category_sims.squeeze(1)  # [bsz, category_mem_size}]
        category_sims = F.softmax(category_sims, dim=2)

        category_mem_vals = self.category_mem_vals.index_select(0, category_idxs)  # [bsz, category_mem_size, mem_dim]
        category_lookup = category_sims.bmm(category_mem_vals).squeeze(1)   # [bsz, mem_dim]

        # combine base and memory through a fc
        result = self.output_fc(torch.cat([base_lookup, category_lookup], dim=1))  # [bsz, output_dim]
        return result


if __name__ == "__main__":

    # Example usage in stroke2instruction model
    bsz = 4
    input_dim = 32
    category_idxs = torch.LongTensor([0, 33, 12, 3])
    mem = SketchMem(input_dim=input_dim, output_dim=input_dim)
    encoded = torch.rand(bsz, input_dim)  # [bsz, dim]
    mem(encoded, category_idxs).size()
