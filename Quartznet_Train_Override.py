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

#global test_array = []
#global test_var 

#global test_array.append(test_var)

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
#import tensorflow as tf
class EncDecCTCModelMultiTest(nemo_asr.models.EncDecCTCModel) :
    


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        #print('\n\n\nCLASS OVERRIDDEN SUCCESSFULLY!!!\n\n\n\n')
        #test_var = test_var + 1
        #test_array.append(test_var)
        #Test_WER()
        #breakpoint()
        #if dataloader_idx > 0 :
        #print('\nbatch_idx: ' + str(batch_idx) + ' dataloader_idx: ' + str(dataloader_idx) + '\n' )
        #val_compare_array[str(dataloader_idx)].append(batch_idx)
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
        
        #breakpoint()
        Test_Step_Dict['val_loss_' + str(dataloader_idx)].append(loss_value.item())
        Test_Step_Dict['val_wer_' + str(dataloader_idx)].append(wer.item())

        if dataloader_idx == val_sets -1 and batch_idx == val_steps -1 :
            #breakpoint()
            #offset = -1
            #print('Offset = -1')
            for key in Test_Epoch_Dict.keys() :
                Test_Epoch_Dict[key].append(np.mean(Test_Step_Dict[key]))
                Test_Step_Dict[key] = []
        """elif dataloader_idx == val_sets -1 and batch_idx == val_steps -2 :
            breakpoint()
            offset = -2
            print('Offset = -2')
        elif dataloader_idx == val_sets -1 and batch_idx == val_steps -3 :
            breakpoint()
            offset = -3
            print('Offset = -3')
        elif dataloader_idx == val_sets -1 and batch_idx == val_steps:
            breakpoint()
            offset = 0
            print('Offset = 0')"""

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

    #model_test = pl.LightningModule

    """def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        train_dataloader=None,  # TODO: remove with 1.6
        ckpt_path: Optional[str] = None,
    ) -> None:
        r""""""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`page <multiple-training-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        #breakpoint()
        #self.model_test = model
        #self.test(model)

        #breakpoint()

    """if train_dataloader is not None:
            rank_zero_deprecation(
                "`trainer.fit(train_dataloader)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.fit(train_dataloaders)` instead. HINT: added 's'"
            )
            train_dataloaders = train_dataloader
        self._call_and_handle_interrupt(
            self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )


    def _test_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        # --------------------
        # SETUP HOOK
        # --------------------
        TrainerMultiTest._log_api_event("test")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.TESTING
        self.state.status = TrainerStatus.RUNNING
        breakpoint()
        self.testing = True

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        # If you supply a datamodule you can't supply test_dataloaders
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.test(dataloaders=..., datamodule=...)`")

        model_provided = model is not None
        model = model or self.lightning_module
        if model is None:
            raise MisconfigurationException(
                "`model` must be provided to `trainer.test()` when it hasn't been passed in a previous run"
            )

        # links data to the trainer
        self._data_connector.attach_data(model, test_dataloaders=dataloaders, datamodule=datamodule)

        self.tested_ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        # run test
        results = self._run(model, ckpt_path=self.tested_ckpt_path)

        assert self.state.stopped
        breakpoint()
        self.testing = False

        return results

    def validate(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
        val_dataloaders=None,  # TODO: remove with 1.6
    ) -> _EVALUATE_OUTPUT:
        r""""""
        Perform one evaluation epoch over the validation set.

        Args:
            model: The model to validate.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to validate.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the validation results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the validation phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end`, etc.
            The length of the list corresponds to the number of validation dataloaders used.
        """
    """ if val_dataloaders is not None:
            rank_zero_deprecation(
                "`trainer.validate(val_dataloaders)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.validate(dataloaders)` instead."
            )
            dataloaders = val_dataloaders
        print('\n\n\n\ TRAINER VALIDATE METHOD OVERRIDEN SUCCESSFULLY!!!\n\n\n')
        breakpoint()
        return self._call_and_handle_interrupt(self._validate_impl, model, dataloaders, ckpt_path, verbose, datamodule)  
    
    @property
    def validate_loop(self) -> EvaluationLoop:
        print('\n\n\nVALIDATION LOOP IN CLASS TRAINER OVERRIDEN!!!\n\n\n\n')
        breakpoint()
        return self._validate_loop

    @property
    def testing(self) -> bool:
        #print('\n\n\nVALIDATION LOOP IN CLASS TRAINER OVERRIDEN!!!\n\n\n\n')
        #breakpoint()
        return self.state.stage == RunningStage.TESTING"""

    """def call_hook""" #### POSSIBILITY

    """def _evaluation_loop(self) -> EvaluationLoop:""" ###Possibility

    """def sanity_checking(self) -> bool:""" #POSSIBILITY

    """def _run_evaluate(self) -> _EVALUATE_OUTPUT:
        #breakpoint()
        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        assert self.evaluating

        # reload dataloaders
        self._evaluation_loop._reload_evaluation_dataloaders()

        # reset trainer on this loop and all child loops in case user connected a custom loop
        self._evaluation_loop.trainer = self

        with self.profiler.profile(f"run_{self.state.stage}_evaluation"), torch.no_grad():
            eval_loop_results = self._evaluation_loop.run()

        # remove the tensors from the eval results
        for result in eval_loop_results:
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu().item()
        #breakpoint()

        return eval_loop_results

    @validate_loop.setter
    def validate_loop(self, loop: EvaluationLoop):"""
    """Attach a custom validation loop to this Trainer.

        It will run with
        :meth:`~pytorch_lighting.trainer.trainer.Trainer.validate`. Note that this loop is different from the one
        running during training inside the :meth:`pytorch_lightning.trainer.trainer.Trainer.fit` call.
        """
    """print('\n\n\nVALIDATION LOOP IN CLASS TRAINER OVERRIDEN!!!\n\n\n\n')
        loop.trainer = self
        self._validate_loop = loop  """

    def save_checkpoint(self, filepath: _PATH, weights_only: bool = False) -> None:
        
        
        #breakpoint()

        

        #self.test(quartznet)
        #thread.join()
        # run the thread
        #thread.run()
        # wait for the thread to finish
        #print('Waiting for the thread...')
        #wer_L, Cor, Sub, Ins, Del, Err, Cor_num, Sub_num, Ins_num, Del_num = wer_ISD('Hello World!', 'Hello World! two')
        #print('\n\nCalculated WER as: ' + str(wer_L)+ '\n\n')
        #thread.join()
        #thread = Thread(target=Test_WER)
        #thread.start()
        self.checkpoint_connector.save_checkpoint(filepath, weights_only)
        #thread.join()

        #self.checkpoint_connector.save_checkpoint(filepath, weights_only)


        """ breakpoint()
        #model = sys.argv[8]
        #test_checkpoint = nemo_asr.models.EncDecCTCModel.restore_from(model, trainer = trainer)
        #test_checkpoint = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)
        #test_checkpoint = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=filepath)#.cuda()
        #test_checkpoint.set_trainer(self)
        
        #breakpoint()
        trainer.test(quartznet)
        
        breakpoint()"""


    



#class FitloopTest(FitLoop):







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

    Test_Step_Dict = {}
    Test_Epoch_Dict = {}
    for i in range(len(multi_val)) :
        Test_Epoch_Dict['val_loss_' + str(i)] = []
        Test_Epoch_Dict['val_wer_' + str(i)] = []
        Test_Step_Dict['val_loss_' + str(i)] = []
        Test_Step_Dict['val_wer_' + str(i)] = []
        val_compare_array[str(i)] = []
        



    #breakpoint()
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

    #breakpoint()
    # trainer_glob = pl.Trainer(gpus=[0], max_epochs=epoch,  enable_progress_bar=True, logger=False)
    #trainer_glob = pl.Trainer(params['trainer'])
    #trainer_glob.resume_from_checkpoint = 'QuartzNet15x5/version_3/checkpoints'
    exp_dir = exp_manager(trainer_glob, params['exp_manager'])

    #breakpoint()

    new_opt = copy.deepcopy(params['model']['optim'])

    print('Learning Rate set to: ', str(sys.argv[2]))
    new_opt['lr'] = float(sys.argv[2])

    #breakpoint()        
    


    if str(sys.argv[5]) == 'English' :
        print('Training from pretrained Model: QuartzNet15x5Base-En')

        model = 'QuartzNet15x5Base-En'
        quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model, trainer = trainer_glob)

    elif str(sys.argv[5]) == 'Spanish' :
        model = 'stt_es_quartznet15x5'
        print('Training from pretrained Model: ' + model)

        model = 'stt_es_quartznet15x5'
        quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model, trainer = trainer_glob)

    elif str(sys.argv[5]) == 'Transfer' :
        model = sys.argv[8]
        #breakpoint()
        quartznet = nemo_asr.models.EncDecCTCModel.restore_from(model, trainer = trainer_glob)
        quartznet.change_vocabulary(
        new_vocabulary= params['model']['labels']
    )

    elif str(sys.argv[5]) == 'Checkpoint' :
        model = sys.argv[8]
        #breakpoint()
        quartznet = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=model).cuda()
        quartznet.set_trainer(trainer_glob)


        

        
    else :
        quartznet = EncDecCTCModelMultiTest(cfg=DictConfig(params['model']), trainer=trainer_glob)

    if  'Checkpoint' not in str(sys.argv[5]) :
        #breakpoint()
        quartznet.setup_optimization(optim_config=DictConfig(new_opt))
        quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
        #quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_multiple_validation_data(val_data_config=params['model']['validation_ds'])
        quartznet.setup_test_data(test_data_config=params['model']['test_ds'])
    #breakpoint()
    #trainer_glob.test(quartznet)
    #breakpoint()
    global thread
    #thread = Thread(target=Test_WER,  daemon=True)#, args=(1.5, 'New message from another thread'))
    #thread.start()
    trainer_glob.fit(quartznet)
    #thread_fit = Thread(target=, args=quartznet,  daemon=True)
    #thread_fit.start()
    #thread_fit.join()
    #thread.join()
    
    #breakpoint()

    write_csv_file('test.csv', Test_Epoch_Dict)

    #print(test_array)
    #breakpoint()
    #test_var = trainer_glob.test(quartznet)
    #breakpoint()




    
    #breakpoint() 
    

    
    new_dir = sys.argv[6]

    if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    model = str(sys.argv[4])
    append = 0
    date.today().isoformat()

    model_name = os.path.join(new_dir, model[: len(model)] + date.today().isoformat() + '_V')


    while(os.path.isfile(model_name + '.nemo')):
        append += 1

    model_name = model_name[:len(model_name)] + str(append)
    model_name += '.nemo'
    quartznet.save_to(model_name)


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
