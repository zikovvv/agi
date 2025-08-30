
from typing import Callable
import gc
import traceback
import weakref
from loguru import logger
import torch

log = logger.info
log_err = logger.error
log_warn = logger.warning
log_debug = logger.debug



def get_cleanup_function(model, optimizer) -> Callable :
    # Use weakref to avoid circular reference issues
    model_ref = weakref.ref(model)
    optimizer_ref = weakref.ref(optimizer)
    def cleanup_model():
        """Clean up CUDA memory allocated by the model"""
        try:
            m = model_ref()
            if m is not None and hasattr(m, 'cpu'):
                m.cpu()
            try : 
                o = optimizer_ref()
                if o is not None and hasattr(o, 'state'):
                    for state in o.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                v.cpu()
            except BaseException :
                traceback.print_exc()
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except:
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    return cleanup_model
