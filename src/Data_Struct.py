import h5py


class DigitStruct:
    def __init__(self, file):
        self.file = h5py.File(file, 'r')
        self.digit_struct_name = self.file['digitStruct']['name']
        self.digit_struct_bbox = self.file['digitStruct']['bbox']

    def get_img_name(self, n):
        '''
            accepts: index for digit structure
            returns: the 'name' string for for the index in digitStruct.
            ie: "2.png"
        '''
        name = ''.join([chr(c[0]) for c in self.file[self.digit_struct_name[n][0]].value])
        return name

    def bbox_helper(self, attribute):
        '''bbox_helper abstracts the bbox or an array of bbox.
           used internally with get_bbox
        '''
        if (len(attribute) > 1):
            attribute = [self.file[attribute.value[j].item()].value[0][0] for j in range(len(attribute))]
        else:
            attribute = [attribute.value[0][0]]
        return attribute

    def get_bbox(self, n):
        '''getBbox returns a dict of data for the n(th) bbox. 
          accepts: index for digit structure
          returns: a hash with the coordiantes for the bbox
          ie {'width': [23.0, 26.0], 'top': [29.0, 25.0], 'label': [2.0, 3.0], 'left': [77.0, 98.0], 'height': [32.0, 32.0]}
        '''
        bbox = {}
        bb = self.digit_struct_bbox[n].item()

        bbox['label'] = self.bbox_helper(self.file[bb]["label"])
        bbox['top'] = self.bbox_helper(self.file[bb]["top"])
        bbox['left'] = self.bbox_helper(self.file[bb]["left"])
        bbox['height'] = self.bbox_helper(self.file[bb]["height"])
        bbox['width'] = self.bbox_helper(self.file[bb]["width"])

        return bbox

    def get_digit_structure(self, n):
        structure = self.get_bbox(n)
        structure['name'] = self.get_img_name(n)
        return structure

    def get_all_imgs_and_digit_structure(self):
        structs = []
        for i in range(len(self.digit_struct_name)):
            structs.append(self.get_digit_structure(i))
        print("Done extract")
        return structs
