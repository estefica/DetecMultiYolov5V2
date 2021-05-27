import os, random,shutil,cv2
def divide_img():

    src_dir = '/content/random/images/'
    train_path ='/content/train/images/'
    dst_dir = ['/content/train/', '/content/valid/','/content/test/']
    files_dst = ['images/','labels/']
    src_labels = '/content/random/labels/'

    for i in dst_dir:
        try:
            os.mkdir(i)
            print(dst_dir)
            for j in files_dst:
                try:
                    os.mkdir(i+j)
                except:
                    pass
        except Exception as e:
            print(e)

    numero_img_random = 3 # numero de imagenes random que escogere por cada imagen del dataset base
    data_setbase = os.listdir(train_path) # LISTA DE IMAGENES TRAIN
    file_list = os.listdir(src_dir) # Lista total de imgs creadas con el factor multiplicativo
    print('Cantidad de donde escoger: '+ str(len(file_list)))

    # TRAIN DATASET
    nombre_imagen = []
    for o_img in data_setbase:
        if o_img.endswith('.jpg'):
            for idx in file_list:
                if o_img.replace('.jpg',"") in idx:
                    nombre_imagen.append(idx)
            for nmbr in range(numero_img_random):
                a = random.choice(nombre_imagen)
                shutil.move(src_dir + a, dst_dir[0] + files_dst[0] + a)
                shutil.move(src_labels + a.replace('.jpg','.txt'),dst_dir[0] + files_dst[1] + a.replace('.jpg','.txt'))
                nombre_imagen.remove(a)
        nombre_imagen = []

    train_n_images = int(len(os.listdir(train_path)))
    eval_n_images = int((15*train_n_images)/80)
    test_n_images = int((5*train_n_images)/80)
    division_imagenes = [eval_n_images, test_n_images]
    contador = 1
    for i in division_imagenes:
        for j in range(i):
            file_list = os.listdir(src_dir)
            a = random.choice(file_list)
            shutil.move(src_dir + a, dst_dir[contador] + files_dst[0] + a)

        random_list = os.listdir(dst_dir[contador]+ files_dst[0])
        for h in random_list:
            fn, ftext = os.path.splitext(h)
            if os.path.exists(src_labels + fn + '.txt'):
                shutil.move(src_labels + fn + '.txt', dst_dir[contador] + files_dst[1] + fn +'.txt')
        contador += 1


def escalado_inicial(path_imagenes,path_imagenes_t,path_save_files,path_save_filest):
 
    for f in os.listdir(path_imagenes):
        if f.endswith('.jpg'):
            
            fn, ftext = os.path.splitext(f)
            
            if os.path.exists(path_imagenes + fn + '.txt'):
                
                img = cv2.imread(path_imagenes + f'{fn}.jpg')
                
                imgResize = cv2.resize(img, (920, 720))

                cv2.imwrite(path_save_files + f'{fn}.jpg', imgResize)

                shutil.copy(path_imagenes_t + fn + '.txt', path_save_filest + f'{fn}.txt')

