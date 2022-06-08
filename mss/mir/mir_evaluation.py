import numpy as np

'''
    dataloader = MIR_DataLoader()
    x_train, y_train = dataloader.load_data(train="test", model="with_postprocessing")

    my_training_batch_generator = My_Custom_Generator(x_train, y_train, 1) #x_train file names
    inp = my_training_batch_generator.__getitem__(80)[0]
    true = my_training_batch_generator.__getitem__(80)[1]

    # print(inp)
    print(true)




    # conv_net.compile(3e-4)
    # len_data = len(x_train)
    # conv_net.train_on_generator(my_training_batch_generator,epochs,len_data) 

    # conv_net.save("first test")
 

    preds  = conv_net.model.predict(inp).tolist() # thisd is predict
    preds = preds[0]
    # print(preds)
    # [preds] = preds # only take first element in batch
    # print(preds)
    from scipy.special import expit
    # print(preds)
    # preds = expit(preds)
    preds = [round(num,2) for num in preds]
    print(preds)
    # print(y_train[0:4])
'''