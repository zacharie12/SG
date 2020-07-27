import torch
import dominate
from dominate.tags import *
import os
from visual_genome import api
from PIL import Image
import numpy

def format_scores(scores, true_index, device):
    score_length = scores.size(0)
    true_tensor = scores[true_index].repeat(score_length)
    binary = torch.ones(score_length).to(device)

    return true_tensor.t(), scores.t(), binary
        
def format_scores_reg(scores, true_index, device):
    target = - torch.ones(scores.size())
    target[true_index] = 1
    target = target.to(device)

    return scores, target

def collate_sgs(sgs, device):
    output = []
    # if sgs is a list of tuples with single elements
    if isinstance(sgs, list):
        for (node, pair, edge) in sgs:
            tensor_pair = pair[0].t().contiguous()
            tensor_pair = tensor_pair.to(device)
            node = node[0].to(device)
            edge = edge[0].to(device)
            output.append((node, tensor_pair, edge))
    # if sgs is a tuple where every element is a big list
    elif isinstance(sgs, tuple):
        nodes, pair_idx, edges = sgs

        for node, pair, edge in zip(nodes, pair_idx, edges):
            tensor_pair = pair.t().contiguous()
            tensor_pair = tensor_pair.to(device)
            node = node.to(device)
            edge = edge.to(device)
            output.append((node, tensor_pair, edge))

    return output


class MistakeSaver():
    def __init__(self, filenames_masked):
        self.filenames_masked = filenames_masked

        self.img_dict = {}
        self.sg_dict = {}

    def add_mistake(self, img_couple, sg_couple, iteration, mistake_type):
        if iteration not in self.img_dict.keys():
            self.img_dict[iteration] = []
            self.sg_dict[iteration] = []

        self.img_dict[iteration].append((img_couple[0], img_couple[1], mistake_type))
        self.sg_dict[iteration].append((sg_couple[0], sg_couple[1], mistake_type))

        return
    
    def toHtml(self, outpath):
        img_dir = 'images'
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        doc = dominate.document(title='Couple Mistakes')
        # get images
        with open(self.filenames_masked, 'r') as filenames:
            lines = filenames.readlines()

            with doc:
                for iteration_number in self.img_dict.keys():
                    h3(str(iteration_number))
                    with table().add(tbody()):
                        for num_couple, (img_couple, sg_couple) in enumerate(zip(self.img_dict[iteration_number], self.sg_dict[iteration_number])):
                            with tr():
                                first_img = os.path.splitext(os.path.basename(lines[img_couple[0]+1]))[0]
                                second_img = os.path.splitext(os.path.basename(lines[img_couple[1]+1]))[0]

                                first_img_link = api.get_image_data(id=first_img).url
                                second_img_link = api.get_image_data(id=second_img).url
                                # HTML img tag
                                with td():
                                    img(src=first_img_link, style="width:224px;height:224px")
                                with td():
                                    img(src=second_img_link, style="width:224px;height:224px")

                                first_sg = self.sg_to_image(sg_couple[0])
                                second_sg = self.sg_to_image(sg_couple[1])

                                # save images
                                first_sg_path = os.path.join(img_dir, f'{iteration_number}_{num_couple}_{0}')
                                first_sg.save(os.path.join(outpath, first_sg_path), 'JPEG')
                                second_sg_path = os.path.join(img_dir, f'{iteration_number}_{num_couple}_{1}')
                                second_sg.save(os.path.join(outpath, second_sg_path), 'JPEG')
                                with td():
                                    img(src=first_sg_path, style="width:224px;height:224px")
                                with td():
                                    img(src=second_sg_path, style="width:224px;height:224px")
                                with td():
                                    p(img_couple[2])

        with open(os.path.join(outpath, 'MistakeCouples.html'), 'w') as outfile:
            outfile.write(str(doc))


    # WE ASSUME RETURN VALUE IS A PIL Image
    def sg_to_image(self, sg):
        return Image.new('RGB', (500, 500))
        pass




    