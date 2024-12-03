import numpy as np
import random
import torchvision.transforms as transforms

class InputHandle:
    def __init__(self, input_param, augmentations=None):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.load()
        
        # Setup augmentations
        self.augmentations = augmentations
        if self.augmentations is not None:
            transformation_list = [transforms.ToPILImage()]
            
            if augmentations['rotate']:
                transformation_list.append(transforms.RandomRotation(30))
            if augmentations['random_flip']:
                transformation_list.append(transforms.RandomHorizontalFlip())
            if augmentations['blur']:
                transformation_list.append(
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
                )
            if augmentations['affine']:
                transformation_list.append(
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
                )
            
            transformation_list.append(transforms.ToTensor())
            self.transform = transforms.Compose(transformation_list)

    def load(self):
        dat_1 = np.load(self.paths[0])
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def total(self):
        return self.data['clips'].shape[0]

    def begin(self, do_shuffle=True):
        self.indices = np.arange(self.total(), dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        
        # Fixed input/output length calculation
        self.current_input_length = 10  # Half of seq_length from data2npz.py
        self.current_output_length = 10  # Half of seq_length from data2npz.py

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        if self.current_position >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            return None
            
        # If this is the last batch and it's incomplete, pad it to match batch_size
        remaining_samples = self.total() - self.current_position
        if remaining_samples < self.minibatch_size:
            self.current_batch_indices = np.pad(
                self.current_batch_indices,
                (0, self.minibatch_size - remaining_samples),
                mode='edge'  # Repeat the last index for padding
            )
        
        # Create batch tensor with shape (batch_size, total_length, height, width, channels)
        batch = np.zeros(
            (self.minibatch_size,  # Always use minibatch_size instead of current_batch_size
             self.current_input_length + self.current_output_length,
             self.data['dims'][0][0],  # height
             self.data['dims'][0][1],  # width
             self.data['dims'][0][2])  # channels
        ).astype(self.input_data_type)
        
        for i in range(self.minibatch_size):  # Always iterate through full batch size
            batch_ind = self.current_batch_indices[i]
            
            # Get input sequence
            input_start = int(self.data['clips'][batch_ind, 0, 0])
            input_length = int(self.data['clips'][batch_ind, 0, 1])
            input_end = input_start + input_length
            input_slice = self.data['input_raw_data'][input_start:input_end]
            
            # Get output sequence
            output_start = int(self.data['clips'][batch_ind, 1, 0])
            output_length = int(self.data['clips'][batch_ind, 1, 1])
            output_end = output_start + output_length
            output_slice = self.data['output_raw_data'][output_start:output_end]
            
            if self.augmentations is not None:
                # Apply augmentations to input sequence
                augmented_input = []
                for frame in input_slice:
                    frame = (frame * 255).astype(np.uint8)
                    frame = self.transform(frame).numpy()
                    augmented_input.append(frame)
                input_slice = np.stack(augmented_input)
                
                # Apply same augmentations to output sequence
                augmented_output = []
                for frame in output_slice:
                    frame = (frame * 255).astype(np.uint8)
                    frame = self.transform(frame).numpy()
                    augmented_output.append(frame)
                output_slice = np.stack(augmented_output)
            
            assert input_slice.shape[0] == self.current_input_length, \
                f"Input slice shape {input_slice.shape} doesn't match expected length {self.current_input_length}"
            assert output_slice.shape[0] == self.current_output_length, \
                f"Output slice shape {output_slice.shape} doesn't match expected length {self.current_output_length}"
            
            # Store sequences in batch tensor
            batch[i, :self.current_input_length] = input_slice
            batch[i, self.current_input_length:] = output_slice
        
        return batch