import os
import torch



def create_blendmarks():
    input_path = '/home/dcor/yaronaloni/Express4D/dataset/COMA_data'
    overwrite = True


    for root, _, files in os.walk(input_path):
        bsps_file = os.path.join(root, 'blendshapes.pt')
        lmks_file = os.path.join(root, 'landmarks_70_centralized.pt')

        if os.path.exists(bsps_file) and os.path.exists(lmks_file):
            bsps = torch.load(bsps_file)
            lmks = torch.load(lmks_file)
            blmks = {'blendshapes': bsps, 'landmarks_70_centralized':lmks}
            torch.save(blmks, os.path.join(root, 'blendmarks_70_centralized.pt'))



if __name__ == '__main__':
    create_blendmarks()