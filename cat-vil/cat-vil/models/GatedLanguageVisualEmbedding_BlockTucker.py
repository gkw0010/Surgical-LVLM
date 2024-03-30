'''
Description     : Gated Language-Vision Embedding module
Paper           : Surgical-VQLA: Transformer with Gated Vision-Language Embedding for 
                  Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Lalithkumar Seenivasan, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git)
                  and GMU (https://github.com/IsaacRodgz/ConcatBERT).
'''

from torch import nn
from transformers import VisualBertConfig
import torch
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim) # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1]<0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j-1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list

def get_chunks(x,sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1,begin,s)
        out.append(y)
        begin += s
    return out

# Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
class VisualBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.hidden_size = 768 # 768 for deit base, 384 for deit small, 192 for deit small
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )
        # self.conv36_visual = torch.nn.Conv1d(in_channels=36, out_channels=25, kernel_size=1)
        # self.conv36_text = torch.nn.Conv1d(in_channels=36, out_channels=25, kernel_size=1)

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        # self.gated_linear = GatedMultimodalLayer(384*25, 384*25, 384*25)
        self.blocktucker_layer = BlockTucker(input_dims = [768*25, 768*25], output_dim=768)
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):

        input_shape = input_ids.size()
        # print('input_shape', input_shape)
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        # print('position_ids', position_ids.shape)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # print('inputs_embeds', inputs_embeds.shape) # torch.Size([64, 25, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print('token_type_embeddings', token_type_embeddings.shape) # torch.Size([64, 25, 768])
        embeddings = inputs_embeds + token_type_embeddings
        # print('embeddings', token_type_embeddings.shape) # torch.Size([64, 25, 768])
        position_embeddings = self.position_embeddings(position_ids)
        # print('position_embeddings', position_embeddings.shape) # torch.Size([1, 25, 768])
        embeddings += position_embeddings
        # print('embeddings', embeddings.shape) # torch.Size([64, 25, 768])

        visual_embeds = self.visual_projection(visual_embeds)
        # print('visual_embeds', visual_embeds.shape) # torch.Size([64, 25, 768])
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        # print('visual_token_type_embeddings', visual_token_type_embeddings.shape) # torch.Size([64, 25, 768])
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        # print('visual_position_ids', visual_position_ids.shape) # torch.Size([64, 25])
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids) 
        # print('visual_position_embeddings', visual_position_embeddings.shape) # torch.Size([64, 25, 768])
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
        # print('visual_embeddings', visual_embeddings.shape) # torch.Size([64, 25, 768])
        # embeddings = torch.flatten(embeddings, start_dim=1, end_dim=-1)
        # visual_embeddings = self.conv36(visual_embeddings)
        # visual_embeddings = torch.flatten(visual_embeddings, start_dim=1, end_dim=-1)        
        # embeddings = self.gated_linear(embeddings, visual_embeddings)  
        # embeddings = torch.reshape(embeddings, (-1, 25, 384))

        # embeddings = torch.cat((embeddings, visual_embeddings), dim=1)
        # print('inputs_embeds', inputs_embeds.shape)
        # print('visual_embeds', visual_embeds.shape)
        # inputs_embeds = self.conv36_text(inputs_embeds)
        # visual_embeds = self.conv36_visual(visual_embeds)
        # print('visual_embeds0.5', visual_embeds.shape)
        # print(inputs_embeds.shape, visual_embeds.shape)
        inputs_embeds = torch.flatten(inputs_embeds, start_dim=1, end_dim=-1)
        visual_embeds = torch.flatten(visual_embeds, start_dim=1, end_dim=-1)
        # print('inputs_embeds1', inputs_embeds.shape)
        # print('visual_embeds1', visual_embeds.shape)        
        embed_sum = [inputs_embeds, visual_embeds]
        
        embeddings = self.blocktucker_layer(embed_sum)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BlockTucker(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1800,
            chunks=10,
            shared=False,
            dropout_input=0.1,
            dropout_pre_lin=0.,
            dropout_output=0.,
            pos_norm='before_cat'):
        super(BlockTucker, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if self.shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)

        self.sizes_list = get_sizes_list(mm_dim, chunks)
        bilinears = []
        for size in self.sizes_list:
            bilinears.append(
                nn.Bilinear(size, size, size)
            )
        self.bilinears = nn.ModuleList(bilinears)
        self.linear_out = nn.Linear(self.mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, bilinear in enumerate(self.bilinears):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            z = bilinear(x0_c, x1_c)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z,p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z