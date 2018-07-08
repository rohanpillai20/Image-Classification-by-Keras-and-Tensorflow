import tensorflow as tf
import numpy as np
import os
import cv2
import time

start = time.time()

try:
    
    # Path of  training images
    train_path = r'C:\Users\T01144\Desktop\T01144\IDLE Scripts\Image Classification\Using Tensorflow\data\train'
    if not os.path.exists(train_path):
        print("No such directory")
        raise Exception
    # Path of testing images
    dir_path = r'C:\Users\T01144\Desktop\T01144\IDLE Scripts\Image Classification\Using Tensorflow\data\alien_test'
    if not os.path.exists(dir_path):
        print("No such directory")
        raise Exception
    
    # Walk though all testing images one by one
    for root, dirs, files in os.walk(dir_path):
        for name in files:

            print("")
            image_path = name
            filename = dir_path +'\\' +image_path
            print(filename)
            image_size=128
            num_channels=3
            images = []
        
            if os.path.exists(filename):
                
                # Reading the image using OpenCV
                image = cv2.imread(filename)
                # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                images.append(image)
                images = np.array(images, dtype=np.uint8)
                images = images.astype('float32')
                images = np.multiply(images, 1.0/255.0) 
            
                # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                x_batch = images.reshape(1, image_size,image_size,num_channels)

                # Let us restore the saved model 
                sess = tf.Session()
                # Step-1: Recreate the network graph. At this step only graph is created.
                saver = tf.train.import_meta_graph('models/trained_model.meta')
                # Step-2: Now let's load the weights saved using the restore method.
                saver.restore(sess, tf.train.latest_checkpoint('./models/'))

                # Accessing the default graph which we have restored
                graph = tf.get_default_graph()

                # Now, let's get hold of the op that we can be processed to get the output.
                # In the original network y_pred is the tensor that is the prediction of the network
                y_pred = graph.get_tensor_by_name("y_pred:0")

                ## Let's feed the images to the input placeholders
                x= graph.get_tensor_by_name("x:0") 
                y_true = graph.get_tensor_by_name("y_true:0") 
                y_test_images = np.zeros((1, len(os.listdir(train_path)))) 


                # Creating the feed_dict that is required to be fed to calculate y_pred 
                feed_dict_testing = {x: x_batch, y_true: y_test_images}
                result=sess.run(y_pred, feed_dict=feed_dict_testing)
                # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                print(result)

                # Convert np.array to list
                a = result[0].tolist()
                r=0

                # Finding the maximum of all outputs
                max1 = max(a)
                index1 = a.index(max1)
                predicted_class = None

                # Walk through directory to find the label of the predicted output
                count = 0
                for root, dirs, files in os.walk(train_path):
                    for name in dirs:
                        if count==index1:
                            predicted_class = name
                        count+=1

                # If the maximum confidence output is largest of all by a big margin then
                # print the class or else print a warning
                for i in a:
                    if i!=max1:
                        if max1-i<i:
                            r=1                           
                if r ==0:
                    print(predicted_class)
                else:
                    print("Could not classify with definite confidence")
                    print("Maybe:",predicted_class)

            # If file does not exist
            else:
                print("File does not exist")
                
except Exception as e:
    print("Exception:",e)

# Calculate execution time
end = time.time()
dur = end-start
print("")
if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
