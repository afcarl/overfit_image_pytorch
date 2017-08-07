import torch.nn as nn
import numpy as np
import torch,os,pickle,time,argparse
from data_loader import * #get_loader,validation_split
from build_vocab import Vocabulary
from build_vocab import build_vocab
from model import  MyModel 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):

    # Create model directory
    full_model_path = args.model_path 

    # Image preprocessing

    transform = transforms.Compose([ 
        transforms.Scale(args.crop_size),
        transforms.ToTensor()])
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load vocabulary wrapper.
    vocab = build_vocab(args.image_dir,1,None)

    # Build data loader
    data_loader = get_loader(args.image_dir,  vocab, transform, args.batch_size, shuffle=True, num_workers=2) 
    data_set = ProcessingDataset(root=args.image_dir, vocab=vocab, transform=transform)
    train_loader = torch.utils.data.DataLoader(data_set,collate_fn=collate_fn)
    train_size = len(train_loader)

    # Build the models
    model = MyModel(args.embed_size, args.hidden_size, len(vocab), vocab)
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            model.train()
            image_ts = to_var(images)
            captions = to_var(captions)
            count = images.size()[0]
            
            # Forward, Backward and Optimize
            model.zero_grad()

            outputs = model(captions, lengths)

            loss = criterion(outputs, image_ts)
            loss.backward()
            optimizer.step()

            correct = outputs.data.eq(image_ts.data).sum()
            accuracy = 100.*correct/count


            # Print log info
            if i % args.log_step == 0:
                #print("i "+str(i))
                #torch.set_printoptions(profile="full")
                for ii,t in enumerate(outputs):
                    result = transforms.ToPILImage()(t.data.cpu())
                    result.save("./results/"+str(i)+"_"+str(ii)+".png")
                    origin = transforms.ToPILImage()(image_ts[ii].data.cpu())
                    origin.save("./results/"+str(i)+"_"+str(ii)+"target.png")
                    with open(("./results/"+str(i)+"_"+str(ii)+"_diff.txt"), 'w') as f:
                        f.write(str(torch.abs(t-image_ts[ii]).sum()))


                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, accuracy: %2.2f Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], accuracy, np.exp(loss.data[0]))) 
                
            # Save the models
            if (i+1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(full_model_path, 'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                           
    torch.save(model.state_dict(), os.path.join(full_model_path, 'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=40 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./training_data/', help='directory for images')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 , help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 , help='dimension of lstm hidden states')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
