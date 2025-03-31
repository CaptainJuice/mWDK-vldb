import argparse
from method import Method
from utils import load_data


def run(args):
    adj_mat, node_features, true_labels = load_data(args.path, args.dataset)
    model = Method(args.dataset,args.algorithm,args.psi,args.t,args.h,)
    embedding = model.get_embedding(adj_mat, node_features)
    # true_labels is just used for evaluation
    model.do_clustering(embedding,args.num_of_clusters,true_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='community detection')
    parser.add_argument('--dataset',default='cora', type=str,help ='datasets')
    parser.add_argument('--path',default = './datasets/real-world datasets/', type=str,help='path of datasets')
    # parser.add_argument('--path',default = '/Users/abin/Desktop/nju/mWDK/dataset/artificial data/', type=str,help='path of datasets')

    parser.add_argument('--psi',default = 64, type=int,help='number of sample points')
    parser.add_argument('--algorithm',default = 'mWDK', type=str,help='mWDK or WDK')
    parser.add_argument('--t',default = 150, type=int,help='number of partitions of IK')
    parser.add_argument('--h',default = 20, type=int,help='number of iterations')
    parser.add_argument('--num_of_clusters',default = 7, type=int,help='number of clusters')

    args = parser.parse_args()
    run2(args=args)


