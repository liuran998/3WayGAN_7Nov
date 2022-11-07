
import torch
from utils import batch_by_size
from torch.autograd import Variable
import logging
from models import *
import torch.nn as nn
from args import args
from tqdm import tqdm

def freeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True



def mrr_mr_hitk(scores, target, x= 1, y = 5, k=10, z =100):
    _, sorted_idx = torch.sort(scores)
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)[0, 0] + 1
    return 1 / target_rank, target_rank, int(target_rank <= x), int(target_rank <= y), int(target_rank <= k), int(target_rank <= z)

# def other_metrics(embedding,valid_data):
#     s = embedding[valid_data[:, 0]].cuda()
#     rel_embed = args.rel_embed
#     r = nn.Embedding.from_pretrained(rel_embed, freeze=True).float().cuda()
#     o = embedding[valid_data[:, 2]].cuda()
#     score = torch.sum(s * r * o, dim=-1)



def calc_mrr( test_data, n_ent, heads, tails, embedding, rel_embed,filt=True):
    mrr_tot = 0
    mr_tot = 0
    hit1_tot = 0
    hit5_tot = 0
    hit10_tot = 0
    hit100_tot = 0
    count = 0
    #Divide the h,r,t into batches , each batchsize=10
    for batch_s, batch_r, batch_t in batch_by_size(args.test_batch_size, *test_data):
        batch_size = batch_s.size(0)
        #Batch size of 10 for h,r,t , each repeated 40943 times per row embedding
        rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
        src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
        dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
        #all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
        #                   .type(torch.LongTensor).cuda(), volatile=True)
        with torch.no_grad():
            #10 rows , each from 0 to 40942
            all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).cuda())   # resolve a warning

        #Average TransE score of h+r-t(all nodes)
        #Average TransE score of h (all nodes) +r-t
        ent_embed = nn.Embedding.from_pretrained(embedding, freeze=True).float().cuda()
        rel_embed = args.rel_embed
        rel_embed = nn.Embedding.from_pretrained(rel_embed, freeze=True).float().cuda()
        batch_dst_scores = score(ent_embed, rel_embed, src_var, rel_var, all_var).data.cuda()
        batch_src_scores = score(ent_embed, rel_embed, all_var, rel_var, dst_var).data.cuda()

        for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
            if filt:

                a = tails[(s.item(), r.item())]
                b = a._nnz()

                if tails[(s.item(), r.item())]._nnz() > 1:
                    # for pythorch 0.4.1 or above
                    c = dst_scores[t]
                    tmp = dst_scores[t].item()
                    dst_scores += tails[(s.item(), r.item())].cuda() * 1e30
                    dst_scores[t] = tmp

                d = heads[(t.item(), r.item())]
                e = d._nnz()

                if heads[(t.item(), r.item())]._nnz() > 1:
                    f = src_scores[s]
                    tmp = src_scores[s].item()
                    src_scores += heads[(t.item(), r.item())].cuda() * 1e30
                    src_scores[s] = tmp

            mrr, mr, hit1, hit5, hit10, hit100 = mrr_mr_hitk(dst_scores, t)
            mrr_tot += mrr
            mr_tot += mr

            hit1_tot += hit1
            hit5_tot += hit5
            hit10_tot += hit10
            hit100_tot += hit100

            mrr, mr, hit1, hit5, hit10, hit100  = mrr_mr_hitk(src_scores, s)
            mrr_tot += mrr
            mr_tot += mr

            hit1_tot += hit1
            hit5_tot += hit5
            hit10_tot += hit10
            hit100_tot += hit100

            count += 2
    # logging.info('Test_MRR=%f, Test_MR=%f, Test_H@1=%f,Test_H@5=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count,
    #              hit1_tot / count, hit5_tot / count, hit10_tot / count)
    tqdm.write("Test_MRR: {}, Test_MR: {}, Test_H@1: {}, Test_H@5: {}, Test_H@10: {}, Test_H@100: {}.".
               format(mrr_tot / count, mr_tot / count, hit1_tot / count, hit5_tot / count, hit10_tot / count, hit100_tot / count))

    return mrr_tot / count

