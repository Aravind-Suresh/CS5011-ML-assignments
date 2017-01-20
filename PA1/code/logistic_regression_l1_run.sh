
declare -a arr=(0.000001 0.00001 0.0001 0.01 0.1 1 10 100 1000 10000)
mkdir ../dataset/DS2/out
for i in "${arr[@]}";
do
./l1_logreg_train -s ../dataset/DS2/data_students/Train_features ../dataset/DS2/data_students/Train_labels "$i" model;
./l1_logreg_classify -t ../dataset/DS2/data_students/Test_labels model ../dataset/DS2/data_students/Test_features "../dataset/DS2/out/res_$i";
done
