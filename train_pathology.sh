#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -l labelled_num in use -g gpus"
   echo -e "\t-l labelled examples in experiments."
   echo -e "\t-g gpus that in use. "
   exit 1
}

while getopts "l:g:" opt; do
  case "$opt" in
    l ) labelled="$OPTARG"
          if ((labelled != 10 && labelled != 20 && labelled != 40 )); then
                 echo "we support the experimental setup for cityscapes as follows:"
                 echo "
    +-------------+------------+------------+------------+
    | hyper-param | 1/8 (10)  | 1/4 (20)  | 1/2 (40) |
    +-------------+------------+------------+------------+
    |    epoch    |     300    |     400    |     500    |
    +-------------+------------+------------+------------+
    |    weight   |     3.0    |     3.0    |     3.0    |
    +-------------+------------+------------+------------+"
              exit 1
        fi
        ;;
    g ) gpus="$OPTARG" ;;
    ? ) helpFunction ;;
  esac
done

if [ "${labelled}" == 20 ]; then
  max_epochs=400
elif [ "${labelled}" == 40 ]; then
  max_epochs=500
else
  max_epochs=300
fi


nohup python3 ./main.py --labeled_examples="${labelled}" --gpus=${gpus} --warm_up=5 --batch_size=4 --semi_p_th=.6 --semi_n_th=.0 \
--epochs=${max_epochs} > pathology_"${labelled}"_unet.out &
