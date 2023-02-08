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
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from threading import Thread

import math

data_dir = '.'


"""
sys.argv[0] == Quartznet_Train.py
sys.argv[1] == epoch
sys.argv[2] == lr
sys.argv[3] == manifest folder
sys.argv[4] == Save_Model_name
sys.argv[5] == pretrained T/F
sys.argv[6] == Output Directory
sys.argv[7] == config file
sys.argv[8] == restore file


"""


from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
#import tensorflow as tf
class EncDecCTCModelMultiTest(nemo_asr.models.EncDecCTCModel) :
    


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
        
        loss_key = 'val_loss_' + key_values_array[dataloader_idx]
        wer_key = 'val_wer_' + key_values_array[dataloader_idx]
        Test_Step_Dict[loss_key].append(loss_value.item())
        Test_Step_Dict[wer_key].append(wer.item())
        if dataloader_idx == val_sets -1 and batch_idx == val_steps -1 :
            
            
            for key in Test_Epoch_Dict.keys() :
                Test_Epoch_Dict[key].append(np.mean(Test_Step_Dict[key]))
                Test_Step_Dict[key] = []
                write_csv_file('test.csv', Test_Epoch_Dict)
        

        #breakpoint()
        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }

from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PATH,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerTypeUnion,
    TRAIN_DATALOADERS,
)
from pytorch_lightning.utilities import (
    _IPU_AVAILABLE,
    _TPU_AVAILABLE,
    device_parser,
    DeviceType,
    DistributedType,
    GradClipAlgorithmType,
    parsing,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
import torch

from pytorch_lightning.utilities.exceptions import ExitGracefullyException, MisconfigurationException

class TrainerMultiTest(pl.Trainer):

    
    def save_checkpoint(self, filepath: _PATH, weights_only: bool = False) -> None:
        
        self.checkpoint_connector.save_checkpoint(filepath, weights_only)
        



def main() : 
    global step_count
    step_count = 0
    epoch = int(sys.argv[1])
    logger = pl.loggers.TensorBoardLogger(
    save_dir=os.getcwd(),
    version=3,
    name='lightning_logs'
    )
    step = 125939 / 16
    global trainer_glob

    #global trainer_glob_test
    global quartznet

    global params
    global Test_Epoch_Dict
    global Test_Step_Dict
    global val_steps
    global val_sets
    global val_compare_array
#    trainer_glob = pl.Trainer(gpus=[0], max_epochs=epoch,  enable_progress_bar=True) 
    trainer_glob = TrainerMultiTest(gpus=[0], max_epochs=epoch, enable_progress_bar=True, logger=False)
    #trainer_glob_test = TrainerMultiTest(gpus=[0], max_epochs=epoch, max_steps=int(step*1.1), enable_progress_bar=True, logger=False)#, checkpoint_callback=True, enable_checkpointing=True)#, logger=False, callbacks=[])#, checkpoint_callback=True)#, logger=False)#, resume_from_checkpoint = 'nemo_experiments/QuartzNet15x5/checkpoints/epoch=4-step=52474.ckpt')
     #, accelerator='ddp')#, auto_scale_batch_size=True)
    
    # REMOVES the ModelCheckpoint object from trainder.callbacks 
    del trainer_glob.callbacks[3]
    #trainer_glob.enable_progress_bar = True

    ### DONT KNOW IF I NEED THIS
    ### https://github.com/NVIDIA/NeMo/issues/1759
    """logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    version=3,
    name='lightning_logs'
)
trainer_glob = pl.Trainer(
    gpus=1, max_epochs=30, precision=16, amp_level='O1', checkpoint_callback=True,         
    resume_from_checkpoint='lightning_logs/version_3/checkpoints/epoch=18-step=68893.ckpt', logger=logger)"""
    
    print('No of Epochs set to: ', str(sys.argv[1]))
    print('Learning Rate set to: ', str(sys.argv[2]))
    print('Manifest Folder set to: ', str(sys.argv[3]))
    print('Model Name set to: ', str(sys.argv[4]))
    print('Pretrained set to: ', str(sys.argv[5]))
    


    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError:
        from ruamel.yaml import YAML
    config_path = sys.argv[7]

    

    
    val_compare_array = {}
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    #trainer_glob = pl.Trainer(**params['trainer'], logger = False)
    
    batch_size= 12

    manifest_fol = str(sys.argv[3])

    train_manifest = data_dir + '/manifests/' + manifest_fol + '/train_manifest.json'
    test_manifest = data_dir + '/manifests/' + manifest_fol + '/test_manifest.json'
    validation_manifest = data_dir + '/manifests/' + manifest_fol + '/valid_manifest.json'

    #multi_val = ['manifests/15_MLSSpanishPlusEngLibri/test_clean_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/test_manifest_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/test_manifest_MLSSpanish.json', 'manifests/15_MLSSpanishPlusEngLibri/test_manifest.json']
    multi_val = ['manifests/0_full_manifest/valid_manifest.json', 'manifests/0_full_manifest/test_manifest.json', 'manifests/15_MLSSpanishPlusEngLibri/test_clean_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/test_manifest_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/test_other_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/valid_manifest_clean_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/valid_manifest_EngLibri.json', 'manifests/15_MLSSpanishPlusEngLibri/valid_manifest_other_EngLibri.json']
    
    val_sets = len(multi_val)
    valid_path = multi_val[-1]
    valid_steps_array = []
    for m in multi_val : 
        with open(m) as valid_json :#
            valid_lines = 0
            for a in valid_json :
                valid_lines += 1
            #steps = math.ceil((train_lines + valid_lines ) / batch_size)

            valid_steps_array.append(math.ceil(valid_lines / batch_size) )
        #one_step = steps 
        #first_step = one_step + 2
    val_steps = valid_steps_array[-1]

    global key_values_array
    key_values_array = []
    for val in multi_val :
        point = val.rfind('.') -1
        slash = val.rfind('/') + 1
        val_key = val[slash:point]
        key_values_array.append(val_key)


    Test_Step_Dict = {}
    Test_Epoch_Dict = {}
    for i in range(len(multi_val)) :
        loss_key = 'val_loss_' + key_values_array[i]
        wer_key = 'val_wer_' + key_values_array[i]
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

     
    #params['model']['train_ds']['shuffle'] = True

    #params['exp_manager']['resume_if_exists'] = True
    #params['exp_manager']['resume_ignore_no_checkpoint'] = True
    #params['exp_manager']['resume_past_end'] = True
    #params['exp_manager']['exp_dir'] = "2023-02-01_18-48-27"
    params['exp_manager']['create_checkpoint_callback'] = True
    
    params['exp_manager']['name'] = 'QuartzNet15x5_3'

    params['trainer']['max_epochs'] = epoch


    exp_dir = exp_manager(trainer_glob, params['exp_manager'])


    new_opt = copy.deepcopy(params['model']['optim'])

    print('Learning Rate set to: ', str(sys.argv[2]))
    new_opt['lr'] = float(sys.argv[2])


    if str(sys.argv[5]) == 'English' :
        print('Training from pretrained Model: QuartzNet15x5Base-En')

        model = 'QuartzNet15x5Base-En'
        quartznet = EncDecCTCModelMultiTest.from_pretrained(model_name=model, trainer = trainer_glob)

    elif str(sys.argv[5]) == 'Spanish' :
        model = 'stt_es_quartznet15x5'
        print('Training from pretrained Model: ' + model)

        model = 'stt_es_quartznet15x5'
        quartznet = EncDecCTCModelMultiTest.from_pretrained(model_name=model, trainer = trainer_glob)

    elif str(sys.argv[5]) == 'Transfer' :
        model = sys.argv[8]
        #breakpoint()
        quartznet = EncDecCTCModelMultiTest.restore_from(model, trainer = trainer_glob)
        quartznet.change_vocabulary(
        new_vocabulary= params['model']['labels']
    )

    elif str(sys.argv[5]) == 'Checkpoint' :
        model = sys.argv[8]
        #breakpoint()
        quartznet = EncDecCTCModelMultiTest.load_from_checkpoint(checkpoint_path=model).cuda()
        quartznet.set_trainer(trainer_glob)


        

        
    else :
        quartznet = EncDecCTCModelMultiTest(cfg=DictConfig(params['model']), trainer=trainer_glob)

    if  'Checkpoint' not in str(sys.argv[5]) :
        quartznet.setup_optimization(optim_config=DictConfig(new_opt))
        quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
        #quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_multiple_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_test_data(test_data_config=params['model']['test_ds'])
  
    trainer_glob.fit(quartznet)
    
    #write_csv_file('test.csv', Test_Epoch_Dict)

    
    

    
    new_dir = sys.argv[6]

    if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    model = str(sys.argv[4])
    append = 0
    date.today().isoformat()

    model_name = os.path.join(new_dir, model[: len(model)] + date.today().isoformat() + '_V')
    csv_name = os.path.join(exp_dir, model[: len(model)] + date.today().isoformat() + '_V')


    while(os.path.isfile(model_name + '.nemo')):
        append += 1

    model_name = model_name[:len(model_name)] + str(append)
    csv_name = csv_name[:len(model_name)] + str(append)
    model_name += '.nemo'
    csv_name += '.csv'
    #quartznet.save_to(model_name)
    write_csv_file(csv_name, Test_Epoch_Dict)


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

def Test_WER():
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
    print(current_line)

def write_csv_file(CSV_Name, Dictionary) :
    with open(CSV_Name, 'w') as output:
            
            for key in Dictionary.keys() :
                output.write(str(key) + ';')
                dict_len = len(Dictionary[key])
                for i in range(0, dict_len) :
                    output.write(str(Dictionary[key][i]) + ';')
                output.write('\n')


if __name__ == '__main__':
    main()
