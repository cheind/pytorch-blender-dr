
def setCameraBackgroundImage(bgImageFolder):
    import os
    import random
    import bpy
    
    # cfg = {'scene.background_images': 'd:/data/val2017/'}
    #for img in bpy.data.images: 
    #    bpy.data.images.remove(img)
    #
    # select random background image from directory of background images
    #bgImageFolder = cfg['scene.background_images']
    bgImageFileNames = os.listdir(bgImageFolder)
    randomIndex = random.choice(range(len(bgImageFileNames)))
    fileName = bgImageFolder + bgImageFileNames[randomIndex]
    print("loading image: " + fileName)
    
    #image = bpy.data.images.load(fileName)
    #bpy.ops.view3d.background_image_add(image)
        
    bpy.data.objects['Camera'].data.background_images.new()
    bpy.data.objects['Camera'].data.background_images[0].image = bpy.data.images.load(fileName)
    bpy.data.objects['Camera'].data.show_background_images = True

bgImageFolder = 'd:/data/val2017/'
setCameraBackgroundImage(bgImageFolder)
