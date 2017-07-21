import tensorflow as tf
import os, sys
from PIL import Image

image_path = 'test_images/leopard2.jpg'

size = (299, 299)

infile = image_path
outfile = os.path.splitext(infile)[0] + '_resized.jpg'
try:
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    old_im_size = im.size
    
    ## By default, black colour would be used as the background for padding!
    new_im = Image.new("RGB", size)

    new_im.paste(im, ((size[0]-old_im_size[0])/2,
	              (size[1]-old_im_size[1])/2))
    
    new_im.save(outfile, "JPEG")
except IOError:
    print "Cannot resize '%s'" % infile



# Read in the image_data
image_data = tf.gfile.FastGFile(outfile, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

init_ops = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_ops)
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
os.remove(outfile)
