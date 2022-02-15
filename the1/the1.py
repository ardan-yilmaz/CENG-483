from PIL import Image
import os
import numpy as np



def chunks(n):
	l =range(0,256)
	n = max(1, n)
	return (l[i:i+n] for i in range(0, len(l), n))

def l1_norm(arr):
    norm = 0
    for i in arr:
        norm += sum(i)
    
    arr = [[i/norm] if i/norm != 0 else [0.000001] for [i] in arr]
    
    return np.array(arr)


def pre__process_for_normalization(arr):
    return np.array([[i] if i != 0 else [0.000001] for i in arr])

def kl_divergence(p, q): 
	#print("p: ", p)
	#print("q: ", q)   

	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def calc_hist_seperately(im, intervals):
    #print("intervals: ", intervals)

    pixel_values = list(im.getdata())

    
    size =0 
    list_of_intervals = []
    for i in intervals:

        #print(i)
        list_of_intervals.append(i)
        size += 1
        
    R_hist = np.zeros(shape=(size,1), dtype=int)
    G_hist = np.zeros(shape=(size,1), dtype=int)
    B_hist = np.zeros(shape=(size,1), dtype=int)      
        

    
    for pixel in pixel_values:
        flag1 =  0
        flag2 =  0
        flag3 =  0

        #print()
        #print("curr pixel: ", pixel)

        for i in range(size):            
            
            #print("curr interval: ", list_of_intervals[i])
            #print()
            #R
            if flag1 == 0 and pixel[0] in list_of_intervals[i]:
                #print("is ", pixel[0], " in range ", list_of_intervals[i])
                R_hist[i][0] += 1
                flag1 = 1
            #G
            if flag2 == 0 and pixel[1] in list_of_intervals[i]:
                #print("is ", pixel[1], " in range ", list_of_intervals[i])
                G_hist[i][0] += 1
                flag2 = 1
            #B
            if flag3 == 0 and pixel[2] in list_of_intervals[i]:
                #print("is ", pixel[2], " in range ", list_of_intervals[i])
                B_hist[i][0] += 1
                flag3 = 1

            if flag1 == 1 and flag2 == 1 and flag3 == 1: break

        #if(flag1 == 0):
            #print(pixel[0], " did not fall into any of the ranges.")
        #if(flag2 == 0):
            #print(pixel[1], " did not fall into any of the ranges.")
        #if(flag3 == 0):
            #print(pixel[2], " did not fall into any of the ranges.")  

            
            
        
    
    return R_hist, G_hist, B_hist

def find_img_per_channel(query, support):

    inc_labeled = 0
    correctly_labeled = 0

    
    for query_key in query.keys():
        
        #print("queried img name: ", query_key)
        
        query_normalized_r = l1_norm(query[query_key]["R"])     
        query_normalized_g = l1_norm(query[query_key]["G"])
        query_normalized_b = l1_norm(query[query_key]["B"])
        min_div = 999999
        
        corresponding_support_name = ""
        for support_key in support.keys():
            #print("checking against: ", support_key)
                                     
            support_normalized_r = l1_norm(support[support_key]["R"])
            support_normalized_g = l1_norm(support[support_key]["G"])
            support_normalized_b = l1_norm(support[support_key]["B"])
            
            


            div_r = kl_divergence(query_normalized_r, support_normalized_r)
            div_g = kl_divergence(query_normalized_g, support_normalized_g)
            div_b = kl_divergence(query_normalized_b, support_normalized_b)
            div = (div_r + div_g + div_b)/3
            #print("calc div: ", div)
            if div < min_div:
                min_div = div
                corresponding_support_name = support_key
            
        
        #print("support img name: ", corresponding_support_name, " with ", min_div)
        if corresponding_support_name == query_key: 
            #print("correctly labeled")
            correctly_labeled += 1
        else: 
            #print("incorrectly labeled with: ", corresponding_support_name)
            inc_labeled += 1
            
        #print()
    
    return correctly_labeled, inc_labeled

def find_img_grid_per_channel(query, support, num_sub_img):
    inc_labeled = 0
    correctly_labeled = 0

    #print("compare time: ", query["Yellow_bellied_Flycatcher_0002_2790772384"][0], "support path: ", support["Yellow_bellied_Flycatcher_0002_2790772384"][0])
    #exit()

    for query_key in query.keys():        
        min_div = 999999
        
        corresponding_support_name = ""
        
        #print("queried img name: ", query_key)
        for support_key in support.keys():
            #print("img tested against: ", support_key)
            
            div_per_comp = 0
            #print("num_sub_img: ", num_sub_img)
            for num in range(num_sub_img):
                query_normalized_r = l1_norm(query[query_key][num]["R"])
                query_normalized_g = l1_norm(query[query_key][num]["G"])
                query_normalized_b = l1_norm(query[query_key][num]["B"])
                
                support_normalized_r = l1_norm(support[support_key][num]["R"])
                support_normalized_g = l1_norm(support[support_key][num]["G"])
                support_normalized_b = l1_norm(support[support_key][num]["B"])
                
                div_r = kl_divergence(query_normalized_r, support_normalized_r)
                div_g = kl_divergence(query_normalized_g, support_normalized_g)
                div_b = kl_divergence(query_normalized_b, support_normalized_b)
                
                div_per_comp += (div_r + div_g + div_b)/3
            
            #take the avg of the sum of divs for a particular comparison b/w a query and support imgs
            div_per_comp = div_per_comp/num_sub_img
            
            #print("kl div calc for ", query_key, " against ", support_key, " is ", div_per_comp)
            #print("curr min div: ", min_div)
            
            # if calc div for this particular comparison is lower, update min and curr label
            if div_per_comp < min_div:
                min_div = div_per_comp
                corresponding_support_name = support_key
            
        
        #print("support img name: ", corresponding_support_name, " with div: ", min_div)
        if corresponding_support_name == query_key: 
            #print("correctly labeled")
            correctly_labeled += 1
        else: 
            #print("incorrectly labeled with: ", corresponding_support_name)
            inc_labeled += 1
            
        #print()
        
    
    return correctly_labeled, inc_labeled

def calc_hist_per_grid(img_data_per_grid, intervals, size):
    
    list_of_intervals = []    
    for i in intervals:
        #print("girrr")
        list_of_intervals.append(i)
        
    #print("list_of_intervals111: ",list_of_intervals)

        
        
    R_hist = np.zeros(shape=(size,1), dtype=int)
    G_hist = np.zeros(shape=(size,1), dtype=int)
    B_hist = np.zeros(shape=(size,1), dtype=int)        
    
    
    for pixel in img_data_per_grid:
        flag1 =  0
        flag2 =  0
        flag3 =  0
        
        #print("curr pixel: ", pixel)
        #print("size: ", size)
        #print("list_of_intervals222: ",list_of_intervals)
        
        for i in range(size): 
            #print("curr interval: ", list_of_intervals[i])

            #R
            if flag1 == 0 and pixel[0] in list_of_intervals[i]:
                #print( pixel[0]," is ", " in range ", list_of_intervals[i],  "so placed in RED channel")
                R_hist[i][0] += 1
                flag1 = 1
            #G
            if flag2 == 0 and pixel[1] in list_of_intervals[i]:
                #print(pixel[1], "is ",  " in range ", list_of_intervals[i], "so placed in GREEN channel")
                G_hist[i][0] += 1
                flag2 = 1
            #B
            if flag3 == 0 and pixel[2] in list_of_intervals[i]:
                #print(pixel[2], "is ",  " in range ", list_of_intervals[i] ,"so placed in BLUE channel")
                B_hist[i][0] += 1
                flag3 = 1


        if flag1 == 0 and flag2 == 0 and flag3 == 0:
            print(pixel, " did not fall into any of ranges :(((  ")
    
    return R_hist, G_hist, B_hist


def calc_hist_3d(p, quantization_interval):    
    return p[0] // quantization_interval, p[1] // quantization_interval, p[2] // quantization_interval 



def find_img_3D_hist(query, support, num_sub_img):
    inc_labeled = 0
    correctly_labeled = 0
    
    
    for query_key in query.keys():
        min_div = 999999
        
        corresponding_support_name = ""
        
        #print("queried img name: ", query_key)
        for support_key in support.keys():
            #print("img tested against: ", support_key)            
                         
            div_per_comp = 0
            #print("num_sub_img: ", num_sub_img)
            for num in range(num_sub_img):  
                #print("index of query[query_key][num] num: ", num)
                query_to_norm = pre__process_for_normalization(query[query_key][num].flatten())
                support_to_norm = pre__process_for_normalization(support[support_key][num].flatten())
                
                query_normalized = l1_norm(query_to_norm)  
                support_normalized = l1_norm(support_to_norm)
                
                div_per_comp += kl_divergence(query_normalized, support_normalized)
            
            #take the avg of the sum of divs for a particular comparison b/w a query and support imgs
            div_per_comp = div_per_comp/num_sub_img
            
            #print("kl div calc for ", query_key, " against ", support_key, " is ", div_per_comp)
            #print("curr min div: ", min_div)
            
            # if calc div for this particular comparison is lower, update min and curr label
            if div_per_comp < min_div:
                #print("new min: ", support_key)
                min_div = div_per_comp
                corresponding_support_name = support_key
        
            
        
        #print("support img name: ", corresponding_support_name, " with div: ", min_div)
        if corresponding_support_name == query_key: 
            #print("correctly labeled")
            correctly_labeled += 1
        else: 
            #print("incorrectly labeled with: ", corresponding_support_name)
            inc_labeled += 1
            
        #print()
        
        
        
        
    
    return correctly_labeled, inc_labeled


#init function is to set the histograms of both the query set specified and the support sets 
def init(mode, grid, quantization_interval, q_path, s_path):

	num_of_intervals = int(256/quantization_interval)

	if mode == "per_channel":
		#print("per_channel mode")
		if grid == 96:
			#print("grid = 96, 1 sub img mode")
			#form query histogram
			per_channel_query = dict()
			for img in os.listdir(q_path):
			    curr_img_path = os.path.join(q_path, img)
			    if not os.path.isfile(curr_img_path): continue
			    im = Image.open(curr_img_path)
			    intervals = chunks(quantization_interval)
			    r, g, b = calc_hist_seperately(im, intervals)
			    #print(r)
			    #print(g)
			    #print(b)    
			    img_name = (img.split('.'))[0]
			    
			    per_channel_query[img_name] = {"R": r, "G":g, "B":b}   

			#form support histogram
			per_channel_support = dict()
			for img in os.listdir(s_path):
			    curr_img_path = os.path.join(s_path, img)
			    if not os.path.isfile(curr_img_path): continue
			    im = Image.open(curr_img_path)
			    intervals = chunks(quantization_interval)
			    r, g, b = calc_hist_seperately(im, intervals)
			    #print(r)
			    #print(g)
			    #print(b)    
			    img_name = (img.split('.'))[0]
			    
			    per_channel_support[img_name] = {"R": r, "G":g, "B":b}

			return per_channel_query, per_channel_support


		#spatial grid
		else:
			#print("q_path: ", s_path)
			
			GRID2 = grid*grid
			number_of_sub_imgs = int(9216/GRID2)
			#print("grid: ", grid, " number_of_sub_imgs: ", number_of_sub_imgs)

			#form query histogram
			per_channel_q1_grid = dict()
			for img in os.listdir(q_path):  
			    img_name = (img.split('.'))[0]
			    
			    per_channel_q1_grid[img_name] = [] 
			    
			    curr_img_path = os.path.join(q_path, img)
			    if not os.path.isfile(curr_img_path): continue
			    im = Image.open(curr_img_path)      
			    pixel_values = np.array(im.getdata())
			    
			    pixel_values = pixel_values.reshape(number_of_sub_imgs,GRID2,3)
			    
			    for p in pixel_values:
			        intervals = chunks(quantization_interval)
			        r, g, b = calc_hist_per_grid(p, intervals, num_of_intervals)   
			        
			        per_channel_q1_grid[img_name].append({"R": r, "G":g, "B":b})


			#form support histogram
			per_channel_support_grid = dict()
			for img in os.listdir(s_path):  
			    img_name = (img.split('.'))[0]
			    
			    per_channel_support_grid[img_name] = [] 
			    
			    curr_img_path = os.path.join(s_path, img)
			    if not os.path.isfile(curr_img_path): continue
			    im = Image.open(curr_img_path)      
			    pixel_values = np.array(im.getdata())
			    
			    pixel_values = pixel_values.reshape(number_of_sub_imgs,GRID2,3)
			    
			    for p in pixel_values:
			        intervals = chunks(quantization_interval)
			        r, g, b = calc_hist_per_grid(p, intervals, num_of_intervals)   
			        
			        per_channel_support_grid[img_name].append({"R": r, "G":g, "B":b})

			return per_channel_q1_grid, per_channel_support_grid        

		





	elif mode == "3d":
		#print("3d hist mode")

		## init histograms for support
		GRID2 = grid*grid
		number_of_sub_imgs = int(9216/GRID2)  
		#print("grid: ", grid, " number_of_sub_imgs: ", number_of_sub_imgs)
		
		support_3D_hist = dict()
		for img in os.listdir(s_path): 
		    img_name = (img.split('.'))[0]       
		    
		    support_3D_hist[img_name] = []
		    
		    curr_img_path = os.path.join(s_path, img)
		    if not os.path.isfile(curr_img_path): continue
		    im = Image.open(curr_img_path)      
		    pixel_values = np.array(im.getdata())    
		    pixel_values = pixel_values.reshape(number_of_sub_imgs,GRID2,3) 	    
		    
		    
		    for i in range(number_of_sub_imgs):
		        per_grid_hist3d = np.zeros(shape=(num_of_intervals,num_of_intervals,num_of_intervals), dtype=int)		        
		        for j in range(GRID2): 
		            #print("curr pixel: ", pixel_values[i][j])
		            #print()
		            intervals = chunks(quantization_interval)
		            #print("curr pixel: ", pixel_values[i][j].shape)
		            coordx, coordy, coordz = calc_hist_3d(pixel_values[i][j], quantization_interval)
		            #print("returned coords: ", coordx, coordy, coordz)
		            per_grid_hist3d[coordx][coordy][coordz] += 1
		            #print("after increment: ", per_grid_hist3d)
		            
		            
		        #print("to append: ", per_grid_hist3d)
		        support_3D_hist[img_name].append(per_grid_hist3d)
		        #print(support_3D_hist)  

		## init histograms for query 

		query_3D_hist = dict()

		for img in os.listdir(q_path): 
		    img_name = (img.split('.'))[0]        
		    
		    query_3D_hist[img_name] = []
		    
		    curr_img_path = os.path.join(q_path, img)
		    if not os.path.isfile(curr_img_path): continue
		    im = Image.open(curr_img_path)      
		    pixel_values = np.array(im.getdata())    
		    pixel_values = pixel_values.reshape(number_of_sub_imgs,GRID2,3)
		    
		    #print("pixel_values.shape: ",pixel_values.shape)
		    #print("pixel_values.shape222: ",pixel_values[0].shape)   
		    
		    
		    for i in range(number_of_sub_imgs):
		        #print("pixels per grid: \n",pixel_values[i].shape)
		        per_grid_hist3d = np.zeros(shape=(num_of_intervals,num_of_intervals,num_of_intervals), dtype=int)
		        
		        for j in range(GRID2): 
		            #print("curr pixel: ", pixel_values[i][j])
		            #print()
		            intervals = chunks(quantization_interval)
		            #print("curr pixel: ", pixel_values[i][j].shape)
		            coordx, coordy, coordz = calc_hist_3d(pixel_values[i][j], quantization_interval)
		            #print("returned coords: ", coordx, coordy, coordz)
		            per_grid_hist3d[coordx][coordy][coordz] += 1
		            #print("after increment: ", per_grid_hist3d)
		            
		            
		        #print("to append: ", per_grid_hist3d)
		        query_3D_hist[img_name].append(per_grid_hist3d)
		        #print(support_3D_hist)		 	

		return query_3D_hist, support_3D_hist		    


	
 

	


if __name__ == "__main__":

	#mode is either "per_channel" or "3d"

	#grid is the pixels in a grid. 
	#grid = 96 FOR A SINGLE GRID
	#grid = 12 FOR 12 x 12 SPATIAL GRID
	#grid = 16 FOR 16 x 16 SPATIAL GRID
	#grid = 24 FOR 24 x 24 SPATIAL GRID
	#grid = 48 FOR 48 x 48 SPATIAL GRID

	#interval is the interval of bins in the histogram. should be able to divide 256, ie, it is an element of the following set: {1,2,4,8,16,32,64,128,256}

	# q_path is the path to query set

	# s_path is the path to support set 

	mode = "3d"

	grid = 24


	quantization_interval = 128
	q_path = os.path.join(os.getcwd(), "query_2") 
	s_path = os.path.join(os.getcwd(), "support_96") 

	#print("mode: ", mode)
	#print("quantization_interval: ", quantization_interval)
	#print("grid: ", grid)
	#print("query: ", q_path)

	query_hist, support_hist = init(mode, grid, quantization_interval, q_path, s_path)
	#print("query_hist")
	#print(query_hist)

	#print("query_hist")


	number_of_sub_imgs = int(9216/(grid*grid))
	#print("number_of_sub_imgs: ", number_of_sub_imgs)


	if mode == "3d":


		correctly_labeled, inc_labeled = find_img_3D_hist(query_hist, support_hist, number_of_sub_imgs)
		
	else:

		if grid == 96:

			correctly_labeled, inc_labeled = find_img_per_channel(query_hist, support_hist)
		else:
			correctly_labeled, inc_labeled = find_img_grid_per_channel(query_hist, support_hist, number_of_sub_imgs)



	print("top-1 accuracy: ", correctly_labeled/(correctly_labeled+inc_labeled))






		

	

    
    







                        
      