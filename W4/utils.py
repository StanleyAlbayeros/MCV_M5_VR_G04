
'''
{'instances': Instances(num_instances=2, image_height=375, image_width=1242, 
fields=[pred_boxes: Boxes(tensor([[551.1287, 181.2309, 576.6088, 200.9708],
        [545.1487, 179.1708, 572.4006, 207.3279]], device='cuda:0')), 
        scores: tensor([0.6186, 0.5270], device='cuda:0'), 
        pred_classes: tensor([0, 0], device='cuda:0')])}
'''
classes = ['Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc', 'DontCare'] 
def output_to_kitti(output,filename):
    # Desired output:
    # class | truncated |
    # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
    kitti_lines = []
    coco_dict = output['instances'].get_fields()
    
    pred_boxes = coco_dict['pred_boxes'].tensor.to("cpu").numpy()
    scores = coco_dict['scores']
    pred_classes = coco_dict['pred_classes']

    with open(filename,"a") as f:
            f.writelines('')
        

    for idx,score in enumerate(scores):
        line = []
        # print(">>>>>>>>>>>>>>")
        # print(output)
        class_idx = pred_classes[idx].to("cpu").item()
        line.append(classes[class_idx])
        line.append('-1')
        line.append('-1')
        line.append('-10')
        
        line.append(str(round(pred_boxes[idx][0],2)))
        line.append(str(round(pred_boxes[idx][1],2)))
        line.append(str(round(pred_boxes[idx][2],2)))
        line.append(str(round(pred_boxes[idx][3],2)))
        line.append('-1')
        line.append('-1')
        line.append('-1')
        line.append('-1000')
        line.append('-1000')
        line.append('-1000')
        line.append('-10')

        line.append(str(round(score.to("cpu").item(),2)))

        line = " ".join(line)

        with open(filename,"a") as f:
            f.writelines(line+'\n')    
        
def create_gt_file():
    
    path = '/home/mcv/datasets/KITTI/training/label_2/'
    
    all = []
    with open('/home/group00/mcv/datasets/KITTI/val_kitti.txt','r') as f:
        all=f.readlines()

    print(all[0])

    with open('/home/group00/working/week2/my_experiment/val_label_dir.txt','w') as f:
        for a in all:
            # print(path+a)
            f.writelines(path+a)


#create_gt_file()