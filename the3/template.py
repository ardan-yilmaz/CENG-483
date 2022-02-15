# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
torch.multiprocessing.set_start_method('spawn', force=True)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 16
max_num_epoch = 100
hps = {'lr':0.05}
#kernel_size = 5
#num_kernels = 3

# ---- options ----
device = 'cuda' if torch.cuda.is_available() else 'cpu' #set device to gpu if avail; else cpu
LOG_DIR = 'checkpoints'
VISUALIZE = 1 # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False







# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset' 
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader


def test_loader(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset' 
    test_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'test_inputs'),device=device)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=0)
    return test_loader




# ---- ConvNet -----
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(1, 16,kernel_size =5, stride=1, padding=2)
    self.conv1_bn = nn.BatchNorm2d(16) # applies batch norm after conv layer 1
    self.conv2 = nn.Conv2d(16, 3, kernel_size= 5, stride=1, padding=2)
    self.conv2_bn = nn.BatchNorm2d(3) # applies batch norm after conv layer 2
    #self.conv4 = nn.Conv2d(8, 3, 5, stride=1, padding=2)


  def forward(self, grayscale_image):
    #layer 1 
    x = self.conv1(grayscale_image)	
    #x = self.conv1_bn(x) 
    
    x = F.relu(x)
    #layer 2
    x = self.conv2(x)
    #x = self.conv2_bn(x)
    """ 

    
    x = F.relu(x)
    #layer 3
    x = self.conv2(x)
    x = F.relu(x) 
    #layer 4
    x = self.conv4(x)
    """

    return x   


# ---- training code -----



def train():	 

	prev_val_loss = None
	print('training begins')
	for epoch in range(max_num_epoch): 
		print("epoch: ", epoch)
		net.train()
		accum_train_loss = 0.0 # training loss of the network
		for iter_train, data in enumerate(train_loader, 0):
			inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.
			optimizer.zero_grad() # zero the parameter gradients
			# do forward, backward, SGD step
			preds = net(inputs)
			loss = criterion(preds, targets)
			loss.backward()
			optimizer.step()
			accum_train_loss += loss.item()
	

		#### VALIDATE every 5 epochs
		if (epoch+1) % 5 == 0:
			all_val_estimations = []
			net.eval()
			accum_val_loss = 0
			val_inputs = None
			val_preds = None
			val_targets = None
			with torch.no_grad():
				for iter_val, val_data in enumerate(val_loader):
					val_inputs, val_targets =  val_data	   
					val_preds = net(val_inputs)
					loss = criterion(val_preds, val_targets)
					accum_val_loss += loss.item()

					if not iter_val:					 
						if not os.path.exists(LOG_DIR):
						    os.makedirs(LOG_DIR)
						#torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint'+ str(epoch) +'.pt'))
						hw3utils.visualize_batch(val_inputs[:5],val_preds[:5],val_targets[:5],os.path.join(LOG_DIR,'example' + str(epoch) + '.png'))					

					#print(val_inputs[:5])					
					#print(val_preds[:5])
					#print(val_targets[:5])

					#append preds, mapped back to [0,255]
					for p in val_preds:
						y = p.detach().detach().cpu().apply_(lambda x: ((x+1) * 127.5)).numpy()
						y = np.transpose(y, (1,2,0))
						all_val_estimations.append(y)

					#print("stop training condition: ", accum_val_loss, " > ", prev_val_loss)
						

			

			#print train/val loss every 5 epochs
			train_loss = accum_train_loss / iter_train
			val_loss = accum_val_loss / iter_val
			train_losses.append(train_loss)
			val_losses.append(val_loss)
			print(f'Epoch = {epoch} | Train Loss = {train_loss:.4f}\tVal Loss = {val_loss:.4f}')	
			#print('Saving the model, end of epoch %d' % (epoch+1))

			if prev_val_loss != None: print("prev_val_loss - accum_val_loss: ", prev_val_loss - accum_val_loss)

			if prev_val_loss == None:
				prev_val_loss = accum_val_loss

			else:
				#check for increase in val loss
				if (prev_val_loss - accum_val_loss) > 10**-5:
					prev_val_loss = accum_val_loss

				# increase in val los, stop training 
				else:
					myfile = open('val_images.txt', 'a')
					data_root = 'ceng483-s19-hw3-dataset'
					val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)

					#save the val img paths
					for iter_val in range(0,2000):
						#print(val_set.imgs[iter_val][0])
						path = val_set.imgs[iter_val][0]
						myfile.write(path)
						myfile.write('\n')							
					myfile.close()	



					return


	# could not early stop, probably learning rate is too low
	#save the val img paths
	print("early stopping failed. probably due to small learning rate")
	for iter_val in range(0,2000):
		#print(val_set.imgs[iter_val][0])
		path = val_set.imgs[iter_val][0]
		myfile.write(path)
		myfile.write('\n')							
	myfile.close()	

	#save the estimated vals to npy file
	np.save('val_estimations',np.asarray(all_val_estimations), allow_pickle=True)






def test():
	myfile = open('test_images.txt', 'a')
	data_root = 'ceng483-s19-hw3-dataset'
	test_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'test_inputs'),device=device)	
	test_estimations = []	
	for _,data in enumerate(test_loader,0):	
		#print(data)	           
		test_preds = net(data[0])
		
		for i, pred in enumerate(test_preds):
			y = pred.detach().detach().cpu().apply_(lambda x: ((x+1) * 127.5)).numpy()
			y = np.transpose(y, (1,2,0))
			test_estimations.append(y)	
			#print(test_set.imgs[i][0])
			path = test_set.imgs[i][0]
			myfile.write(path)
			myfile.write('\n')
		break

	myfile.close()
	np.save('estimations_test',np.asarray(test_estimations), allow_pickle=True)
		
	



	

if __name__ == "__main__":
	val_batch_size = 100 #num of imgs to be loaded for val/test
	print('device: ' + str(device))
	net = Net().to(device=device)
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
	train_loader, val_loader = get_loaders(batch_size,device)

	#global vars to be used to draw the train/val loss curves
	train_losses= []
	val_losses = []
	


	train()	
	print('Finished Training')

	#TEST
	test_loader = test_loader(val_batch_size,device)
	test()

	"""
	plt.figure(figsize=(10,5))
	plt.title("Training and Validation Loss")
	plt.plot(val_losses,label="val")
	plt.plot(train_losses,label="train")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()	
	"""

	


