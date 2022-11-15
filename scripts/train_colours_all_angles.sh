python src/main.py \
dataset_path=/home/andrej/datasets/leddartech \
data.preprocessed_path=colour_all_angles \
tasks=['preprocess','train'] \
devices=[1] \
data.materials=["black","blue","orange","white","yellow","unknown"] \
preprocess.max_angle_in_deg=-1 \
hydra.run.dir=outputs/tcn/colour_all_angles \
