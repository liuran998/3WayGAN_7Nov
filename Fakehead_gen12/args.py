import argparse

parser = argparse.ArgumentParser(description='RGCN')

#Overall
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--evaluate-every", type=int, default=1)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument('--test_batch_size', type=int, default=2)
parser.add_argument('--node_embed_size', type=int, default=100)
#parser.add_argument("--trigger_times", type=int, default=20)

parser.add_argument("--freeze_epoch", type=int, default=100000)

#RGCN
parser.add_argument("--graph-split-size", type=float, default=0.5)
parser.add_argument("--negative-sample", type=int, default=10)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--n-bases", type=int, default=4)
parser.add_argument("--regularization", type=float, default=1e-2) #Shared by gen and rgcn
parser.add_argument("--rgcn_lr", type=float, default=1e-2)#Shared by gen and rgcn

#Dis
parser.add_argument("--graph-batch-size", type=int, default=300)  # 30000
parser.add_argument("--dis_lr", type=float, default=0.0001)
parser.add_argument("--epoch_d", type=int, default=1)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

#Gen
parser.add_argument('--label_smooth', type=float, default=0)
# parser.add_argument("--gen_ratio", type=float, default=0.1) #[ 0.1 , 0.5 , 1]
# parser.add_argument('--lambda_gen', type=float, default=1e-5, help='Lambda in Generator Loss')
parser.add_argument("--grad-norm", type=float, default=1.0)
parser.add_argument("--gen_fake_ratio", type=int, default=5)
parser.add_argument("--epoch_g", type=int, default=5)
parser.add_argument("--gen_lr", type=float, default=0.001)


args = parser.parse_args()
