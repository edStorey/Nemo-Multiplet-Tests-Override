import sys
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
sys.path.insert(1, '../WER_Function/')
from WER_Function import wer_ISD
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig
import pytorch_lightning as pl
import json
import numpy as np
import os
import copy
from datetime import date

import math

data_dir = '.'


"""
sys.argv[0] == Quartznet_Train_Override.py
sys.argv[1] == epoch
sys.argv[2] == lr
sys.argv[3] == training manifest folder
sys.argv[4] == validation manifests folder
sys.argv[5] == Experiment Name
sys.argv[6] == pretrained T/F
sys.argv[7] == config file
sys.argv[8] == restore file
sys.argv[9] == if this exists in a valid filename ignore it in first test run


"""


def main() : 

    ## Define Global variables
    
    global Test_Epoch_Dict
    global Test_Step_Dict
    global val_steps
    global val_sets
    global val_compare_array
    global csv_exp_dir
    global key_values_array
    global step_counter
    global epoch_count
    global sanity_check


    sanity_check = 0
    epoch_count = 0
    step_counter = 0
    epoch = int(sys.argv[1])

    ## Define trainer from TrainerMultiTest Child Class]
    debug_gpus = [0]
    boole_gpus = [0, 1]
    gpus = debug_gpus
    trainer_Multi = pl.Trainer(gpus=gpus, max_epochs=epoch, enable_progress_bar=True, logger=False) #, accelerator='ddp')
    del trainer_Multi.callbacks[3]
    
    print('No of Epochs set to: ', str(sys.argv[1]))
    print('Learning Rate set to: ', str(sys.argv[2]))
    print('Train Manifest Folder set to: ', str(sys.argv[3]))
    print('Validation Manifest Folder set to: ', str(sys.argv[4]))
    print('Experimenty Name set to: ', str(sys.argv[4]))
    print('Pretrained set to: ', str(sys.argv[5]))
    print('Config File set to: ', str(sys.argv[6]))
    print('Rextore From File set to: ', str(sys.argv[6]))

 
    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError:
        from ruamel.yaml import YAML
    config_path = sys.argv[7]

    step_count = 0
    batch_size= 12
 
    ## Load params from config.yaml file    
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    
    
    manifest_fol = str(sys.argv[3])

    train_manifest = data_dir +  manifest_fol + '/train_manifest.json'
    test_manifest = data_dir +  manifest_fol + '/test_manifest.json'
    #validation_manifest = data_dir + '/manifests/' + manifest_fol + '/valid_manifest.json'
    validation_top_dir = data_dir + str(sys.argv[4]) 

    ## Detect all files that do npot have train in the nam e
    multi_val = [validation_top_dir + '/' + f for f in os.listdir(validation_top_dir) if os.path.isfile(os.path.join(validation_top_dir, f)) and not 'train' in f]
    val_sets = len(multi_val)

    ## Calculate steps per epoch for each validation files
    valid_steps_array = []
    for m in multi_val : 
        with open(m) as valid_json :#
            valid_lines = 0
            for a in valid_json :
                valid_lines += 1
            #breakpoint()
            valid_steps_array.append(math.ceil((valid_lines / batch_size)/len(gpus)))

    val_steps = valid_steps_array[-1]

    print('\n\nNUMBER OF STEPS PER EPOCH: ' + str(val_steps) + '\n\n')

    ### Define Keys for below Dicts based on input file names
    key_values_array = []
    for val in multi_val :
        point = val.rfind('.') 
        slash = val.rfind('/') + 1
        val_key = val[slash:point]
        key_values_array.append(val_key)

    ### Set up loss and WER recording dictionaries
    Test_Step_Dict = {}
    Test_Epoch_Dict = {}
    val_compare_array = {}
    Test_Epoch_Dict['Epoch'] = []
    for i in range(len(multi_val)) :
        loss_key = key_values_array[i] + '_loss'
        wer_key = key_values_array[i] + '_wer'
        Test_Epoch_Dict[loss_key] = []
        Test_Epoch_Dict[wer_key] = []
        Test_Step_Dict[loss_key] = []
        Test_Step_Dict[wer_key] = []
        val_compare_array[str(i)] = []
        



    params['model']['train_ds']['max_duration'] = 60.0
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    #params['model']['validation_ds']['manifest_filepath'] = validation_manifest
    params['model']['validation_ds']['manifest_filepath'] = multi_val
    params['model']['test_ds']['manifest_filepath'] = test_manifest
    params['model']['train_ds']['num_workers'] = 16
    params['model']['train_ds']['batch_size'] = batch_size
    params['model']['validation_ds']['batch_size'] = batch_size
    params['model']['test_ds']['batch_size'] = batch_size

     
    """#params['model']['train_ds']['shuffle'] = True
    params['exp_manager']['resume_if_exists'] = True
    #params['exp_manager']['resume_ignore_no_checkpoint'] = True
    #params['exp_manager']['resume_past_end'] = True
    #params['exp_manager']['exp_dir'] = "2023-02-01_18-48-27"""""
    

    params['exp_manager']['create_checkpoint_callback'] = True
    #params['exp_manager']['name'] = 'QuartzNet15x5_3'
    params['exp_manager']['name'] = str(sys.argv[5])
    params['trainer']['max_epochs'] = epoch


    exp_dir = exp_manager(trainer_Multi, params['exp_manager'])
    csv_exp_dir = exp_dir


    new_opt = copy.deepcopy(params['model']['optim'])
    new_opt['lr'] = float(sys.argv[2])


    ### Choose what form of training to do
    if str(sys.argv[6]) == 'English' :
        print('Training from pretrained Model: QuartzNet15x5Base-En')

        model = 'QuartzNet15x5Base-En'
        quartznet = EncDecCTCModelMultiTest.from_pretrained(model_name=model, trainer = trainer_Multi)

    elif str(sys.argv[6]) == 'Spanish' :
        model = 'stt_es_quartznet15x5'
        print('Training from pretrained Model: ' + model)

        model = 'stt_es_quartznet15x5'
        quartznet = EncDecCTCModelMultiTest.from_pretrained(model_name=model, trainer = trainer_Multi)

    elif str(sys.argv[6]) == 'Transfer' :
        model = sys.argv[8]
        #breakpoint()
        quartznet = EncDecCTCModelMultiTest.restore_from(model, trainer = trainer_Multi)
        quartznet.setup_optimization(optim_config=DictConfig(new_opt))
        quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
        #quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_multiple_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_test_data(test_data_config=params['model']['test_ds'])

        epoch_count = -2

        avoid_key_word = sys.argv[9]

        Test_Epoch_Dict['Epoch'].append(epoch_count)
        for t in range(len(multi_val)) :

            loss_key = key_values_array[t] + '_loss'
            wer_key = key_values_array[t] + '_wer'
            params['model']['test_ds']['manifest_filepath']= multi_val[t]
            quartznet.setup_test_data(params['model']['test_ds'])
            #breakpoint()
            if avoid_key_word not in  key_values_array[t] :
                test_var = trainer_Multi.test(quartznet) ###, verbose = False)
                test_string = 'For file' + str(t) +' : ' + str(test_var) + '\n'
                Test_Epoch_Dict[loss_key].append(test_var[0]['test_loss'])
                Test_Epoch_Dict[wer_key].append(test_var[0]['test_wer'])
            else :
                Test_Epoch_Dict[loss_key].append('N/A')
                Test_Epoch_Dict[wer_key].append('N/A')


        epoch_count += 1
        breakpoint()
        quartznet.change_vocabulary(
        new_vocabulary= params['model']['labels']
        )

        Test_Epoch_Dict['Epoch'].append(epoch_count)
        for t in range(len(multi_val)) :
            loss_key = key_values_array[t] + '_loss'
            wer_key = key_values_array[t] + '_wer'
            params['model']['test_ds']['manifest_filepath']= multi_val[t]
            quartznet.setup_test_data(params['model']['test_ds'])
            
            test_var = trainer_Multi.test(quartznet) ###, verbose = False)
            test_string = 'For file' + str(t) +' : ' + str(test_var) + '\n'
            print(test_string)

            """tests_array += test_string
            key = t[:len(t)-json_len]
            key_loss = key + '_loss'
            key_wer = key + '_wer'"""
            #breakpoint()
            Test_Epoch_Dict[loss_key].append(test_var[0]['test_loss'])
            Test_Epoch_Dict[wer_key].append(test_var[0]['test_wer'])
        epoch_count += 1
    

    elif str(sys.argv[6]) == 'Checkpoint' :
        model = sys.argv[8]
        #breakpoint()
        quartznet = EncDecCTCModelMultiTest.load_from_checkpoint(checkpoint_path=model).cuda()
        quartznet.set_trainer(trainer_Multi)
   
    else :
        quartznet = EncDecCTCModelMultiTest(cfg=DictConfig(params['model']), trainer=trainer_Multi)


    if  'Checkpoint' not in str(sys.argv[6]) :
        quartznet.setup_optimization(optim_config=DictConfig(new_opt))
        quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
        #quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_multiple_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_test_data(test_data_config=params['model']['test_ds'])
    
    breakpoint()
    trainer_Multi.fit(quartznet)
    #breakpoint()

    #print('\n\nNumber of steps found in final val set: ' + str(step_counter) + '\n\n')
    #print('\n\nNumber of steps calculated in final val set: ' + str(valid_steps_array[-1]) + '\n\n')
    
    model = sys.argv[5]
    append = 0
    date.today().isoformat()

    #model_name = os.path.join(new_dir, model[: len(model)] + date.today().isoformat() + '_V')
    csv_name = os.path.join(exp_dir, model + date.today().isoformat() + '_V')


    while(os.path.isfile(csv_name + '.csv')):
        append += 1

    #model_name = model_name[:len(model_name)] + str(append)
    csv_name = csv_name[:len(csv_name)] + str(append)
    #model_name += '.nemo'
    csv_name += '.csv'
    #quartznet.save_to(model_name)
    write_csv_file(csv_name, Test_Epoch_Dict)



"""Child Class for nemo_asr.models.EncDecCTCModel editing validation step method"""
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
class EncDecCTCModelMultiTest(nemo_asr.models.EncDecCTCModel) :


    
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        global step_counter
        global epoch_count
        global sanity_check

        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        self._wer.update(
            predictions=predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()
        if sanity_check == 1 :
            loss_key = key_values_array[dataloader_idx] + '_loss'
            wer_key = key_values_array[dataloader_idx] + '_wer'
            Test_Step_Dict[loss_key].append(loss_value.item())
            Test_Step_Dict[wer_key].append(wer.item())
        #breakpoint()
        if  (step_counter == 2 and sanity_check == 0) :
            sanity_check = 1
        if dataloader_idx == val_sets -1 :
            #breakpoint()
            step_counter += 1
            #print(step_counter)
            if batch_idx == val_steps -1 :
                Test_Epoch_Dict['Epoch'].append(epoch_count)#str(#))
                for key in Test_Epoch_Dict.keys() :
                    if 'Epoch' not in key :
                        Test_Epoch_Dict[key].append(np.mean(Test_Step_Dict[key]))
                        Test_Step_Dict[key] = []
                        csv_name = 'current_losswer_values.csv'
                        write_csv_file(os.path.join(csv_exp_dir, csv_name), Test_Epoch_Dict)
                sanity_check = 1    
                epoch_count += 1
        
        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }


"""Child Class for pl.Trainer editing save_checkpoint method"""
"""from pytorch_lightning.utilities.types import _PATH
class TrainerMultiTest(pl.Trainer):
    
    def save_checkpoint(self, filepath: _PATH, weights_only: bool = False) -> None:       
        self.checkpoint_connector.save_checkpoint(filepath, weights_only)"""


"""Child Class for object"""
class NoStdStreams(object):
        def __init__(self,stdout = None, stderr = None):
            self.devnull = open(os.devnull,'w')
            self._stdout = stdout or self.devnull or sys.stdout
            self._stderr = stderr or self.devnull or sys.stderr

        def __enter__(self):
            self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
            self.old_stdout.flush(); self.old_stderr.flush()
            sys.stdout, sys.stderr = self._stdout, self._stderr

        def __exit__(self, exc_type, exc_value, traceback):
            self._stdout.flush(); self._stderr.flush()
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            self.devnull.close()


"""Write Dictionary to .csv file"""
def write_csv_file(CSV_Name, Dictionary) :
    with open(CSV_Name, 'w') as output:
            
            for key in Dictionary.keys() :
                output.write(str(key) + ';')
                dict_len = len(Dictionary[key])
                for i in range(0, dict_len) :
                    output.write(str(Dictionary[key][i]) + ';')
                output.write('\n')




#### Unused function, however may be needed in other places
"""def Test_WER():
    wer_full = []
    with open('manifests/2_Only_US/valid_manifest.json') as json_file :
        for a in json_file :
            test = json.loads(a)

            file_path = [test['audio_filepath']]
            with NoStdStreams() :
                #GPU
                transcription = quartznet.cuda().transcribe(paths2audio_files=file_path)
                

            text = [test['text']]

            wer_L, Cor, Sub, Ins, Del, Err, Cor_num, Sub_num, Ins_num, Del_num = wer_ISD(text[0], transcription[0])#


            wer_full.append(round(wer_L, 3))
    full_WER = np.mean(wer_full)
    current_line = '\n\n2_Only_US/valid_manifest.json' + ' WER: ' + str(full_WER) + '\n\n'
    print(current_line)"""


if __name__ == '__main__':
    main()
