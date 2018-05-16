#!/bin/bash

# while IFS='' read -r line || [[ -n "$line" ]]; do
#     echo "Experiment: $line"
# done < "$1"


EXPNAME="experiment_unet_checker_0001"
PROJECT="netruns"
DIRIP=fernandez@learn.cambia.caltech.edu
HOMESERVER="/home/fernandez/code/datasciencebowl/python"
NAMEMODEL="chk000009.pth.tar"

#model_best.pth.tar
#chk000000.pth.tar

eval "$(ssh-agent -s)"
ssh-add /home/pdmf/.ssh/caltechkeyos.pri

mkdir ./$PROJECT/$EXPNAME

# # # # copy log    
# # echo ">>copy log from server"
# # echo ">>$HOMESERVER/$PROJECT/$EXPNAME/$EXPNAME.log  ==> ./$PROJECT/$EXPNAME/" 
# # scp  "$DIRIP:$HOMESERVER/$PROJECT/$EXPNAME/$EXPNAME.log" "./$PROJECT/$EXPNAME/"


# echo ">>parser"
# python parse_log_manifold.py \
# "./$PROJECT/$EXPNAME/$EXPNAME.log" \
# "./$PROJECT/$EXPNAME/" 

#copy model
echo ">>copy models from server"
echo ">>$HOMESERVER/$PROJECT/$EXPNAME/models/$NAMEMODEL  ==> ./$PROJECT/$EXPNAME/"
scp "$DIRIP:$HOMESERVER/$PROJECT/$EXPNAME/models/$NAMEMODEL" "./$PROJECT/$EXPNAME/"

# echo ">>copy solution from server"
# echo ">>$HOMESERVER/$PROJECT/$EXPNAME/models/$NAMEMODEL  ==> ./$PROJECT/$EXPNAME/"
# scp "$DIRIP:$HOMESERVER/$PROJECT/submission.csv" "./$PROJECT/$EXPNAME/"


exit 
EOF
