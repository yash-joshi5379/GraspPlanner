class CaptureStdout:
    """
    Captures standard output messages and emits signals for each write.

    This class redirects stdout output through a buffer and emits complete lines
    via a Qt signal, allowing real-time display of printed output in a GUI.
    """

    def __init__(self, progress_signal):
        """
        Initialize the CaptureStdout instance.

        Args:
            progress_signal: A Qt signal object that will be emitted with captured text.
        """
        self.progress_signal = progress_signal
        self.buffer = ""

    def write(self, text):
        """
        Write text to the buffer and emit complete lines via the progress signal.

        This method buffers incoming text and emits complete lines (those ending with
        a newline character) through the progress_signal. Incomplete lines are kept
        in the buffer for the next write call.

        Args:
            text (str): The text to write to the buffer.
        """
        self.buffer += text
        # Emit when we have a complete line (ends with newline)
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Emit all complete lines
            for line in lines[:-1]:
                if line.strip():
                    self.progress_signal.emit(line + '\n')
            # Keep the incomplete line in buffer
            self.buffer = lines[-1]
