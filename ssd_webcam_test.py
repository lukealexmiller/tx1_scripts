import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import caffe
import sort
import time
import cv2

# Toggle display.
display = True

if display:
    colours = np.random.rand(32,3)
    #plt.ion()
    fig = plt.figure() 

# Specify number of iterations
test_iters = 10

# Use GPU at location 0
caffe.set_device(0)
caffe.set_mode_gpu()

# Load model and weights
net = caffe.Net('/home/ubuntu/ssd-tx1/test.prototxt',
                '/home/ubuntu/ssd-tx1/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel',
                caffe.TEST)

# Create instance of the SORT tracker
mot_tracker = sort.Sort()


for i in range(test_iters):
    i += 1
    print("Iteration number: %d"%i)

    # Perform inference
    out = net.forward()

    # Detection output
    # out (list) FORMAT: [image_id, label, confidence, xmin, ymin, xmax, ymax]
    dets = out['detection_out'][0][0]
    #dets = np.asarray(dets)

    # Convert from [xmin, ymin, xmax, ymax] to [x, y, w, h]
    dets[:,5:7] -= dets[:,3:5]
    print(np.shape(dets))
    # Tracker input
    # dets_shaped (numpy array) FORMAT: [[x,y,w,h,score],[x,y,w,h,score],...]

    print(np.shape(dets[:,3:]))
    print(np.shape(dets[:,:3]))
    dets_shaped = np.hstack((dets[:,3:],dets[:,:3]))
    print(np.shape(dets_shaped))
    
    if(display):
    	# Extract input image
    	im = net.blobs['data'].data
    	im = np.transpose(im[0],axes=[1,2,0])
        print(np.shape(im))
        #im = cv2.imread('/home/ubuntu/Downloads/cat.jpg')
        print(np.shape(im))
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.imshow(im)
        plt.title('Tracked Targets')

    # Initialize timing
    total_time = 0.0
    start_time = time.time()
    trackers = mot_tracker.update(dets)
    cycle_time = time.time() - start_time
    total_time += cycle_time

    for d in trackers:
        #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(i,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
        if(display):
            d = d.astype(np.uint32)
            #ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
            #ax1.set_adjustable('box-forced')

    if(display):
	print('enter')
        fig.canvas.flush_events()
        plt.show()
        ax1.cla()
    
    time.sleep(10)
    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,i,i/total_time))
