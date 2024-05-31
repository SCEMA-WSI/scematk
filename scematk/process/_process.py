from abc import ABC, abstractmethod
from typing import List

from ..image._image import Image


class Process(ABC):
    def __init__(self, name: str) -> None:
        """Constructor for Process class

        Args:
            name (str): Name of the process
        """
        assert isinstance(name, str), "name must be a string"
        self.name = name

    @abstractmethod
    def run(self, image: Image) -> Image:
        """Run the process on the image

        Args:
            image (Image): A SCEMATK Image object

        Returns:
            Image: A SCEMATK Image object
        """
        pass


class Processor:
    def __init__(self, processes: List[Process] | None = None) -> None:
        """Constructor for Processor class

        Args:
            processes (List[Process] | None, optional): A list of SCEMATK Process objects. Defaults to None.
        """
        assert processes is None or isinstance(
            processes, list
        ), "processes must be a list of Process objects"
        assert processes is None or all(
            isinstance(proc, Process) for proc in processes
        ), "processes must be a list of Process objects"
        if processes is None:
            self.processes = []
        else:
            self.processes = processes

    def add_process(self, process: Process) -> None:
        """Add a process to the processor

        Args:
            process (Process): A SCEMATK Process object
        """
        assert isinstance(process, Process), "process must be a Process"
        self.processes.append(process)

    def run(self, image: Image) -> Image:
        """Run all processes on the image

        Args:
            image (Image): A SCEMATK Image object

        Returns:
            Image: A SCEMATK Image object
        """
        assert isinstance(image, Image), "image must be an Image"
        image = image
        for process in self.processes:
            image = process.run(image)
        return image

    def __repr__(self) -> str:
        """String representation of the Processor object

        Returns:
            str: String representation of the Processor object
        """
        if len(self.processes) == 0:
            ret_str = "Empty processor object"
        else:
            ret_str = "Processor object with processes:"
            for i, proc in enumerate(self.processes):
                ret_str += f"\n\t({i+1}) {proc.name}"
        return ret_str

    def _repr_html_(self) -> str:
        """HTML representation of the Processor object

        Returns:
            str: HTML representation of the Processor object
        """
        total_width = 400
        html = f' <div style="width: {total_width}px; background-color: #202020; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">'
        html += "<h1>SCEMATK Processor Object</h1>"
        if len(self.processes) == 0:
            html += "<p>Empty processor object</p>"
        else:
            html += "<p>Processor object with processes:</p>"
            for i, proc in enumerate(self.processes):
                html += f"<p>({i+1}) {proc.name}</p>"
        html += "</div>"
        return html
