import subprocess
import os
import platform
from typing import Any, List, Mapping, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

class CustomLlamaCLI(LLM):
    """
    Custom LLM class that interfaces with a local llama.cpp CLI executable.

    It automatically locates the llama-cli executable in './libs/'
    and the .gguf model in './models/' relative to the script's directory.
    """
    n_predict: int = 128
    threads: int = 2
    ctx_size: int = 2048
    temperature: float = 0.8
    conversation: bool = False

    _executable_path: Optional[str] = None
    _model_path: Optional[str] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Determine executable name based on OS
        if platform.system() == "Windows":
            self._executable_path = os.path.join(script_directory, "libs", "llama-cli.exe")
        else:
            self._executable_path = os.path.join(script_directory, "libs", "llama-cli")

        if not os.path.exists(self._executable_path):
            raise FileNotFoundError(
                f"Error: Llama CLI executable not found at {self._executable_path}. "
                "Please ensure it's in the './libs/' directory relative to your script."
            )

        # Find the .gguf model file
        models_dir = os.path.join(script_directory, "models")
        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"Error: Models directory not found at {models_dir}. "
                "Please create it and place your .gguf model inside."
            )

        found_model = False
        for filename in os.listdir(models_dir):
            if filename.lower().endswith(".gguf"):
                self._model_path = os.path.join(models_dir, filename)
                found_model = True
                break

        if not found_model:
            raise FileNotFoundError(
                f"Error: No .gguf model found in {models_dir}. "
                "Please place at least one .gguf model file in this directory."
            )
            
    @property
    def _llm_type(self) -> str:
        return "custom_llama_cli"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Runs the llama.cpp CLI with the given prompt and parameters.
        """
        command = [
            self._executable_path,
            "-m", self._model_path,
            "-n", str(self.n_predict),
            "-t", str(self.threads),
            "-p", prompt,
            "-ngl", "0",
            "-c", str(self.ctx_size),
            "--temp", str(self.temperature),
            "-b", "1",
        ]
        if self.conversation:
            command.append("-cnv")

        # Add stop sequences if provided (llama.cpp supports --reverse-prompt)
        if stop is not None:
            for s in stop:
                command.extend(["--reverse-prompt", s])

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            output = process.stdout.strip()

            # Llama.cpp often echoes the prompt. We need to remove it.
            # This is a common issue with llama.cpp CLI and requires careful handling.
            # A more robust solution might involve parsing the output line by line.
            if output.startswith(prompt):
                output = output[len(prompt):].strip()
            
            # Further truncate if any stop sequence was explicitly added
            if stop is not None:
                for s in stop:
                    if s in output:
                        output = output.split(s)[0].strip()
                        break
            
            if process.stderr:
                # Optionally log stderr if needed, but don't return it as part of the output
                if run_manager:
                    run_manager.on_llm_new_token(f"STDERR from llama-cli: {process.stderr}", verbose=True)

            return output
        except subprocess.CalledProcessError as e:
            error_message = (
                f"Error running llama-cli: {e}\n"
                f"Command: {' '.join(e.cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"Output (stdout): {e.output}\n"
                f"Standard Error (stderr): {e.stderr}"
            )
            if run_manager:
                run_manager.on_llm_error(e)
            raise RuntimeError(error_message) from e
        except FileNotFoundError as e:
            error_message = (
                f"Error: Llama CLI executable not found. "
                f"Please ensure '{self._executable_path}' exists and is executable."
            )
            if run_manager:
                run_manager.on_llm_error(e)
            raise FileNotFoundError(error_message) from e
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            if run_manager:
                run_manager.on_llm_error(e)
            raise RuntimeError(error_message) from e

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Return the identifying parameters.
        """
        return {
            "executable_path": self._executable_path,
            "model_path": self._model_path,
            "n_predict": self.n_predict,
            "threads": self.threads,
            "ctx_size": self.ctx_size,
            "temperature": self.temperature,
            "conversation": self.conversation,
        }
