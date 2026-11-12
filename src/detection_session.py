"""Keep native detection crashes outside the batch/report process."""

import multiprocessing


class DetectionFailure(RuntimeError):
    """The detector is unavailable; scene-only analysis can continue."""


def _worker(connection):
    from .workflow import select_subject

    try:
        while True:
            image = connection.recv()
            if image is None:
                return
            try:
                connection.send({'subject': select_subject(image)})
            except Exception as error:
                connection.send({'error': f'{type(error).__name__}: {error}'})
    except EOFError:
        return
    finally:
        connection.close()


class DetectionSession:
    """Reuse models in one spawned worker; disable detection after a worker failure."""

    def __init__(self, timeout=90):
        self.timeout = timeout
        self.process = None
        self.connection = None
        self.failure = None

    def __enter__(self):
        return self

    def __call__(self, image):
        if self.failure:
            raise DetectionFailure(self.failure)
        try:
            if self.process is None:
                context = multiprocessing.get_context('spawn')
                self.connection, child = context.Pipe()
                self.process = context.Process(target=_worker, args=(child,), daemon=True)
                try:
                    self.process.start()
                finally:
                    child.close()
            self.connection.send(image)
            if not self.connection.poll(self.timeout):
                raise TimeoutError(f'检测超过 {self.timeout} 秒')
            reply = self.connection.recv()
            if 'error' in reply:
                raise DetectionFailure(reply['error'])
            return reply['subject']
        except (EOFError, BrokenPipeError, OSError, TimeoutError, DetectionFailure) as error:
            if self.process is not None and self.process.pid is not None:
                self.process.join(timeout=0.1)
            self.failure = str(error) or f'检测进程异常退出，退出码 {self.process.exitcode}'
            self.close()
            print(f'检测已停用，后续使用全图评分：{self.failure}', flush=True)
            raise DetectionFailure(self.failure) from error

    def close(self):
        if self.connection is not None:
            self.connection.close()
        if self.process is not None and self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)

    def __exit__(self, *_):
        self.close()
