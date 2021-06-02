import os, random, shutil, cv2
import lab_multis


def divide_img(data_setbase):
    src_dir = '/content/random/images/'
    train_path = '/content/train/images/'
    valid_path =  '/content/valid/images/'
    test_path = '/content/test/images/'
    dst_dir = ['/content/train/', '/content/valid/', '/content/test/']
    files_dst = ['images/', 'labels/']
    src_labels = '/content/random/labels/'
    menos_check = 0
    porcent_train = 75
    porcent_valid = 20
    porcent_test = 5

    numero_img_random = 2  # numero de imagenes random que escogere por cada imagen del dataset base
      # LISTA DE IMAGENES TRAIN------->>> !!!!!!!!!!!!!!!!check it
    file_list = os.listdir(src_dir)  # Lista total de imgs creadas con el factor multiplicativo
    print('Cantidad de donde escoger: ' + str(len(file_list)))#numero de imagenes en random imagenes

    # TRAIN DATASET
    nombre_imagen = []
    try:
      for o_img in data_setbase:
          if o_img.endswith('.jpg'):
              for idx in file_list:
                  if o_img.replace('.jpg', "") in idx:
                      nombre_imagen.append(idx)
              for nmbr in range(numero_img_random):
                  a = random.choice(nombre_imagen)
                  shutil.move(src_dir + a, dst_dir[0] + files_dst[0] + a)
                  shutil.move(src_labels + a.replace('.jpg', '.txt'),
                              dst_dir[0] + files_dst[1] + a.replace('.jpg', '.txt'))
                  nombre_imagen.remove(a)
          nombre_imagen = []
      train_n_images = int(len(os.listdir(train_path)))
      eval_n_images = int((porcent_valid * train_n_images) / porcent_train) - int(len(os.listdir(valid_path)))
      test_n_images = int((porcent_test * train_n_images) / porcent_train) -- int(len(os.listdir(test_path)))
      division_imagenes = [eval_n_images, test_n_images]
      contador = 1
      for i in division_imagenes:
          for j in range(i):
              file_list = os.listdir(src_dir)
              a = random.choice(file_list)
              shutil.move(src_dir + a, dst_dir[contador] + files_dst[0] + a)

          random_list = os.listdir(dst_dir[contador] + files_dst[0])
          for h in random_list:
              fn, ftext = os.path.splitext(h)
              if os.path.exists(src_labels + fn + '.txt'):
                  shutil.move(src_labels + fn + '.txt', dst_dir[contador] + files_dst[1] + fn + '.txt')
          contador += 1
    except:
      pass

def escalado_inicial(fn, path_imagenes_t, path_save_files, path_save_filest,img,w_deseado,h_deseado): #nombre, y random_files random_txt, imagen
    imgResize = cv2.resize(img, (w_deseado, h_deseado))
    cv2.imwrite(path_save_files + f'{fn}.jpg', imgResize)
    shutil.copy(path_imagenes_t + fn + '.txt', path_save_filest + f'{fn}.txt')

