python src/main.py \
dataset_path=/home/andrej/datasets/leddartech \
data.preprocessed_path=colours_0_angles \
tasks=['preprocess','train'] \
devices=[0] \
data.materials=["black","blue","orange","white","yellow","unknown"] \
preprocess.max_angle_in_deg=10 \
hydra.run.dir=outputs/tcn/colours_0_angles \
