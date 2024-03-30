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
        self.mfh_layer = MFH(input_dims = [768*25, 768*25], output_dim=768)
    
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
        # visual_embeddings = self.conv36(visual_embeddings)
        # print('visual_embeds0.5', visual_embeds.shape)
        # inputs_embeds = self.conv36_text(inputs_embeds)
        # visual_embeds = self.conv36_visual(visual_embeds)
        
        inputs_embeds = torch.flatten(inputs_embeds, start_dim=1, end_dim=-1)
        visual_embeds = torch.flatten(visual_embeds, start_dim=1, end_dim=-1)
        # print('inputs_embeds1', inputs_embeds.shape)
        # print('visual_embeds1', visual_embeds.shape)        
        embed_sum = [inputs_embeds, visual_embeds]
        embeddings = self.mfh_layer(embed_sum)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MFH(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=768,
            factor=5,
            activ_input='relu',
            activ_output='relu',
            normalize=True,
            dropout_input=0.1,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(MFH, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear_out = nn.Linear(mm_dim*2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0_0(x[0])
        x1 = self.linear1_0(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_0_skip = x0 * x1

        if self.dropout_pre_lin:
            z_0_skip = F.dropout(z_0_skip, p=self.dropout_pre_lin, training=self.training)

        z_0 = z_0_skip.view(z_0_skip.size(0), self.mm_dim, self.factor)
        z_0 = z_0.sum(2)

        if self.normalize:
            z_0 = torch.sqrt(F.relu(z_0)) - torch.sqrt(F.relu(-z_0))
            z_0 = F.normalize(z_0, p=2)

        #
        x0 = self.linear0_1(x[0])
        x1 = self.linear1_1(x[1])

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)

        z_1 = x0 * x1 * z_0_skip

        if self.dropout_pre_lin > 0:
            z_1 = F.dropout(z_1, p=self.dropout_pre_lin, training=self.training)

        z_1 = z_1.view(z_1.size(0), self.mm_dim, self.factor)
        z_1 = z_1.sum(2)

        if self.normalize:
            z_1 = torch.sqrt(F.relu(z_1)) - torch.sqrt(F.relu(-z_1))
            z_1 = F.normalize(z_1, p=2)

        #
        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)

        if self.activ_output:
            z = getattr(F, self.activ_output)(z)

        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z