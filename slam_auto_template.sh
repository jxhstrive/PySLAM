#! /bin/bash

downloadBucket=""
uploadBucket=""
dataset=""
cloudType="cloud"
tmp_dir="tmp"
codeBase="PySLAM"

# Download data

python lidar_dataset.py --config_dir cloud.yaml --cloud_type ${cloudType} --bucket ${downloadBucket} --prefix ${dataset} --dir_path ../${tmp_dir}

# Preprocess data, merge

cp merge.sh ../${tmp_dir}/merge.sh
cd ../${tmp_dir} && chmod +x merge.sh
./merge.sh ${dataset}
cd ../${codeBase}

# SLAM

python main_icp_slam.py --data_base_dir ../${tmp_dir}/${dataset}/sweeps --num_icp_points 40000

# Create map

for map in `ls result/${dataset}/map*`; do
    if [ -d result/${dataset}/${map} ]; then
        python PcAlign.py --camera_calib_dir intrinsic_extrinsic --slam_paras_file_dir result/${dataset}/${map}
    fi
done

# Upload map data

python lidar_dataset.py --config_dir cloud.yaml --cloud_type ${cloudType} --bucket ${uploadBucket} --prefix ${dataset} --dir_path result_pcd/${dataset}

# Remove raw data

# cp delete_tmp.sh ../${tmp_dir}/delete_tmp.sh
# cd ../${tmp_dir} && chmod +x delete_tmp.sh
# ./delete_tmp.sh ${dataset}
# cd ../${codeBase}

