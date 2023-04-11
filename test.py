# import pickle as p
# from Concepts.ConceptBank import ConceptBank
# import matplotlib.pyplot as plt
# import os
#
# from Concepts.helper import load_images_from_files
#
# with open('saved_concept_bank.pkl', 'rb') as file:
#     concept_bank = p.load(file)
#
# CB = ConceptBank(concept_bank['concept_bank_dct']['mixed8'])
#
# fig = CB.plot_concept('userDefined_africanEar2.jpg', num_images=5, shape=(90, 90))
# fig.show()
# #
# # fig = CB.plot_concept('userDefined_africanEar1.jpg', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('userDefined_trunk1.jpg', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('userDefined_tusk1.jpg', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('African_elephant__concept13', num_images=5, shape=(90, 90))
# # fig.show()
# #
# # fig = CB.plot_concept('African_elephant__concept5', num_images=5, shape=(90, 90))
# # fig.show()
#
# # class_folder = './questionnaire/indian_ear'
# # img_filenames = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]
# # imgs = load_images_from_files(filenames=img_filenames, max_imgs=20, num_workers=1)[:5]
# # fig, axs = plt.subplots(1, 5)
# # axs = axs.flatten()
# # for idx, img, ax in zip(range(5), imgs, axs):
# #     ax.imshow(img)
# #     ax.axis('off')
# # fig.tight_layout()
# # fig.savefig(fname=f'subset_images/images_indian_ear.png', dpi=400)

import os
import tarfile

tarfiles = os.listdir('data/ImageNet2')
tarfiles = [os.path.join('data/ImageNet2', file) for file in tarfiles]

for file in tarfiles:
    # Open the .tar file
    tar = tarfile.open(file, "r")

    # Extract all the contents of the .tar file to a directory
    tar.extractall(path=f'{file.rstrip(".tar")}/')

    # Close the .tar file
    tar.close()

    os.remove(file)
