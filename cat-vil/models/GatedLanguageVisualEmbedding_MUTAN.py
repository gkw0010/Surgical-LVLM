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
        self.mutan_layer = Mutan(input_dims = [768*25, 768*25], output_dim=768)
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
        embeddings = self.mutan_layer(embed_sum)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
# class GatedMultimodalLayer(nn.Module):
#     def __init__(self, size_in1, size_in2, size_out):
#         super().__init__()
#         self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

#         # Weights hidden state modality 1
#         weights_hidden1 = torch.Tensor(size_out, size_in1)
#         self.weights_hidden1 = nn.Parameter(weights_hidden1)

#         # Weights hidden state modality 2
#         weights_hidden2 = torch.Tensor(size_out, size_in2)
#         self.weights_hidden2 = nn.Parameter(weights_hidden2)

#         # Weight for sigmoid
#         weight_sigmoid = torch.Tensor(size_out*2)
#         self.weight_sigmoid = nn.Parameter(weight_sigmoid)

#         # initialize weights
#         nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

#         # Activation functions
#         self.tanh_f = nn.Tanh()
#         self.sigmoid_f = nn.Sigmoid()

#     def forward(self, x1, x2):
#         h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
#         h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
#         x = torch.cat((h1, h2), dim=1)
#         z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

#         return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2


class Mutan(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=700,
            rank=10,
            shared=False,
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        super(Mutan, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = 10
        self.output_dim = output_dim
        self.dropout_input = 0.1
        self.dropout_pre_lin = 0.
        self.dropout_output = dropout_output
        self.normalize = normalize
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim) 
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim*rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim) 
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim*rank)
        self.linear_out = nn.Linear(mm_dim, output_dim) 
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])

        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        # print('x0', x0.shape) # torch.Size([64, 768])
        # print('x1', x1.shape) # torch.Size([64, 768])
        m0 = self.merge_linear0(x0)
        m1 = self.merge_linear1(x1)
        # print('m0', m0.shape) # torch.Size([64, 7680])
        # print('m1', m1.shape) # torch.Size([64, 7680])
        m = m0 * m1
        # print('m', m.shape) # torch.Size([64, 7680])
        m = m.view(-1, self.rank, self.mm_dim)
        # print('m', m.shape) # torch.Size([64, 10, 768])
        z = torch.sum(m, 1)
        # print('z1', z.shape) # torch.Size([64, 768])
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        # print('z2', z.shape)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        # print('z3', z.shape)
        z = self.linear_out(z)
        # print('z4', z.shape)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        # print('z5', z.shape) # torch.Size([64, 768])
        return z