#!/bin/bash

# 0.001 to large!
learning_rates=(0.00005 0.0001 0.0005 0.001)
negative_samples=(50 100 200)
lambda=(0.0001 0.0005)
epoch=(50 70 90)

output_dir="./hyperparameter_output"
mkdir -p $output_dir

for lr in "${learning_rates[@]}"
do
    job_name="model_lr${lr}"
    sbatch_script="$output_dir/sbatch_$job_name.sh"

    echo "#!/bin/bash" >> $sbatch_script
    echo "#SBATCH --partition=ashton" >> $sbatch_script
    echo "#SBACTH --qos=ashton" >> $sbatch_script
    echo "#SBATCH --job-name=$job_name" >> $sbatch_script
    echo "#SBATCH --output=$output_dir/$job_name.%j.out" >> $sbatch_script
    echo "#SBATCH --time=02:00:00" >> $sbatch_script
    echo "#SBATCH --mem=4G" >> $sbatch_script
    echo "#SBATCH --gres=gpu:1" >> $sbatch_script

    echo "conda init" >> $sbatch_script
    echo "source ~/miniconda3/etc/profile.d/conda.sh" >> $sbatch_script
    echo "conda activate kge" >> $sbatch_script
    echo "python -u /h/224/yfsun/KGE_implementation_Distmult/execute.py -lr $lr -n 50 -l 0.0001 --validation -e 60 &> $output_dir/$job_name.out" >> $sbatch_script

done

# for ep in "${epoch[@]}"
# do
#     job_name="model_epoch${ep}"
#     sbatch_script="$output_dir/sbatch_$job_name.sh"

#     echo "#!/bin/bash" >> $sbatch_script
#     echo "#SBATCH --partition=ashton" >> $sbatch_script
#     echo "#SBACTH --qos=ashton" >> $sbatch_script
#     echo "#SBATCH --job-name=$job_name" >> $sbatch_script
#     echo "#SBATCH --output=$output_dir/$job_name.%j.out" >> $sbatch_script
#     echo "#SBATCH --error=$output_dir/$job_name.%j.err" >> $sbatch_script
#     echo "#SBATCH --time=02:00:00" >> $sbatch_script
#     echo "#SBATCH --mem=4G" >> $sbatch_script

#     echo "conda init" >> $sbatch_script
#     echo "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 pytorch -c nvidia" >> $sbatch_script
#     echo "python /h/224/yfsun/KGE_implementation_Distmult/execute.py -lr $lr -n $neg -l $lambda --validation -e $ep &> $output_dir/$job_name.out" >> $sbatch_script

#     sbatch $sbatch_script
# done

# for lam in "${lambda[@]}"
# do  
#     job_name="model_lam${lam}"
#     sbatch_script="$output_dir/sbatch_$job_name.sh"

#     echo "#!/bin/bash" >> $sbatch_script
#     echo "#SBATCH --partition=ashton" >> $sbatch_script
#     echo "#SBACTH --qos=ashton" >> $sbatch_script
#     echo "#SBATCH --job-name=$job_name" >> $sbatch_script
#     echo "#SBATCH --output=$output_dir/$job_name.%j.out" >> $sbatch_script
#     echo "#SBATCH --error=$output_dir/$job_name.%j.err" >> $sbatch_script
#     echo "#SBATCH --time=02:00:00" >> $sbatch_script
#     echo "#SBATCH --mem=4G" >> $sbatch_script

#     echo "conda init" >> $sbatch_script
#     echo "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 pytorch -c nvidia" >> $sbatch_script
#     echo "python /h/224/yfsun/KGE_implementation_Distmult/execute.py -lr $lr -n $neg -l $lambda --validation -e $ep &> $output_dir/$job_name.out" >> $sbatch_script

#     sbatch $sbatch_script
# done

# for neg in "${negative_samples[@]}"
# do
#     job_name="model_neg${neg}"
#     sbatch_script="$output_dir/sbatch_$job_name.sh"

#     echo "#!/bin/bash" >> $sbatch_script
#     echo "#SBATCH --partition=ashton" >> $sbatch_script
#     echo "#SBACTH --qos=ashton" >> $sbatch_script
#     echo "#SBATCH --job-name=$job_name" >> $sbatch_script
#     echo "#SBATCH --output=$output_dir/$job_name.%j.out" >> $sbatch_script
#     echo "#SBATCH --error=$output_dir/$job_name.%j.err" >> $sbatch_script
#     echo "#SBATCH --time=02:00:00" >> $sbatch_script
#     echo "#SBATCH --mem=4G" >> $sbatch_script

#     echo "conda init" >> $sbatch_script
#     echo "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 pytorch -c nvidia" >> $sbatch_script
#     echo "python /h/224/yfsun/KGE_implementation_Distmult/execute.py -lr $lr -n $neg -l $lambda --validation -e $ep &> $output_dir/$job_name.out" >> $sbatch_script

#     sbatch $sbatch_script
# done