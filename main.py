import os
import argparse

from models import Multigrid

flags = argparse.ArgumentParser(description='Initializing the Multigrid system')

flags.add_argument('--prefetch', default=True, help='True for prefetch images')
flags.add_argument('--dataset_name', default='rock', help='Folder address')
flags.add_argument('--input_pattern', default='.jpg', help='Image extension')
flags.add_argument('--sample_dir', default='sample', help='Sample directory')
flags.add_argument('--checkpoint_dir', default='checkpoint', help='Directory for checkpoints')
flags.add_argument('--T', default=90, help='Langevin Dynamics steps', type=int)
flags.add_argument('--batch_size', default=100, help='Batch size')
flags.add_argument('--epochs', default=700, help='Number of Epochs to train for')
flags.add_argument('--image_size', default=64, help='Image size of training images')
flags.add_argument('--num_threads', default=2, help='Number of threads for reading images')
flags.add_argument('--read_len', default=100, help='Number of batches per reading')
flags.add_argument('--delta', default=0.3, help='Langevin step size', type=float)
flags.add_argument('--learning_rate', default=0.3, help='Learning rate', type=float)
flags.add_argument('--beta1', default=0.5, help='Momentum')
flags.add_argument('--weight_decay', default=0.0001, help='Weight decay')
flags.add_argument('--ref_sig', default=50, help='Std. deviation for Gaussian distribution')
flags.add_argument('--clip_grad', default=1.0, help='Clipping gradient')
flags.add_argument('--load_models', default=False, help='Loading models')
flags.add_argument('--model_dir', default='models', help='For Loading models')
flags.add_argument('--epoch_file', default='epochs.txt', help='No. of Epochs in a file')
flags.add_argument('--scale_list', default=[1, 4, 16, 64], help="Scale list")

FLAGS = flags.parse_args()

def main():
       
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
        
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
        
    try:
        file = open(FLAGS.epoch_file, "r")
        e = int(file.read())
        file.close()
        if e > 0:
            FLAGS.load_models = True
            
    except IOError:
        file = open(FLAGS.epoch_file, 'w')
        file.write(str(0))
        file.close()
        
    return Multigrid(FLAGS)
    
if __name__ == '__main__':
    Multigrid = main()
    Multigrid.train()
