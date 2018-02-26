class Config:
    data_path = "C:\\Users\\Administrator\\Desktop\\XYL2RGB"
    model_path = "" #/home/hadoop/GalaxyGAN_python/images_blur/checkpoint/model_10.ckpt' 
    output_path = "C:\\Users\\Administrator\\Desktop\\XYL2RGB"

    img_size = 500
    adjust_size = 500
    train_size = 424
    img_channel = 3
    conv_channel_base = 64
    
    batch_size = 14

    #learning_rate = 0.0001
    beta1 = 0.5
    max_epoch = 10000000
    L1_lambda = 100
    save_per_epoch=1
