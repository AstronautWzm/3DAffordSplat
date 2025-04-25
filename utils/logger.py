import logging


def logger_init(rank,state,setting,is_resume):

    logger = logging.getLogger('affordsplat')
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    stream_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

    if rank == 0:

        if is_resume:
            log_path = "log/"+state+'_'+setting+"_resume.log"
        else:
            log_path = "log/"+state+'_'+setting+".log"
        file_handler = logging.FileHandler(log_path,'w')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logger.info('logger started')

    return logger

