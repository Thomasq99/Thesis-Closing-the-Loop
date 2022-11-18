import h5py
from ACE import ACE
from tensorflow.keras import datasets, layers, models
from ACE_helper import *
import os
import shutil

# START Variables
TRAIN = False
data_prepared = True
model_path = './models/3dshapes_model'
class_to_id = {'cube': 0, 'cylinder': 1, 'sphere': 2, 'tictac': 3}
target_class = 'sphere'
working_dir = f'ACE_output/3dshapes_{target_class}'
bottlenecks = ['conv2d_3']
source_dir = './data/ACE_3dshapes_toy'
num_random_concepts = 20

# related DIRs on CNS to store results #######
discovered_concepts_dir = os.path.join(working_dir, 'concepts/')
results_dir = os.path.join(working_dir, 'results/')
cavs_dir = os.path.join(working_dir, 'cavs/')
activations_dir = os.path.join(working_dir, 'acts/')
results_summaries_dir = os.path.join(working_dir, 'results_summaries/')
if os.path.exists(working_dir):
    shutil.rmtree(working_dir)
os.makedirs(working_dir)
os.makedirs(discovered_concepts_dir)
os.makedirs(results_dir)
os.makedirs(cavs_dir)
os.makedirs(activations_dir)
os.makedirs(results_summaries_dir)
random_concept = 'random_discovery'  # Random concept for statistical testing

# Create Model:
if TRAIN:
    with h5py.File('./data/3dshapes_toy.h5', 'r') as data:
        X_train = np.array(data['X_train'])
        X_test = np.array(data['X_test'])
        y_train = np.array(data['y_train'])
        y_test = np.array(data['y_test'])

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('started fitting model')

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test), batch_size=64)
    model.save(model_path)
else:
    model = tf.keras.models.load_model(model_path)

# prepare data for ACE
if not data_prepared:
    with h5py.File('./data/3dshapes_toy.h5', 'r') as data:
        X_train = np.array(data['X_train'])
        y_test = np.array(data['y_test'])

    ace_create_source_dir_from_array(X_train, y_train, target_class, source_dir, class_to_id,
                                     num_random_concepts=num_random_concepts, max_imgs_target_class=500, ow=False)

# run ACE
ace = ACE(model, bottlenecks, target_class, source_dir,
          activations_dir, cavs_dir, random_concept, class_to_id,
          num_random_concepts=num_random_concepts, num_workers=100)

# create patches
print("Creating patches")
ace.create_patches_for_data()
image_dir = os.path.join(discovered_concepts_dir, 'images')
os.makedirs(image_dir)
save_images(image_dir, (ace.discovery_images * 255).astype(np.uint8))  # save images used for creating patches

# Discovering Concepts
print('discovering concepts')
ace.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
del ace.dataset  # Free memory
del ace.image_numbers
del ace.patches

# Save discovered concept images (resized and original sized)
save_concepts(ace, discovered_concepts_dir)

# Calculating CAVs and TCAV scores
print('calculating CAVs and TCAVs')
cav_accuracies = ace.cavs(min_acc=0.0)
scores = ace.tcavs(test=False)
save_ace_report(ace, cav_accuracies, scores, results_summaries_dir + 'ace_results.txt')

print('Started plotting concepts')
# Plot examples of discovered concepts
for bottleneck in ace.bottlenecks:
    plot_concepts(ace, bottleneck, 10, address=results_dir)
