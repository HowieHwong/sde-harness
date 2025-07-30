"""
Base CLI class for SDE-Harness projects.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Dict, Any


class CLIBase(ABC):
    """
    Base class for project CLI interfaces.
    
    Provides standard command-line interface patterns for projects.
    """
    
    def __init__(self, project_name: str):
        """
        Initialize the CLI.
        
        Args:
            project_name: Name of the project
        """
        self.project_name = project_name
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description=f"{self.project_name} - SDE-Harness Project"
        )
        
        # Common arguments for all projects
        parser.add_argument(
            "--models-file",
            default="config/models.yaml",
            help="Path to models configuration file"
        )
        parser.add_argument(
            "--credentials-file", 
            default="config/credentials.yaml",
            help="Path to credentials configuration file"
        )
        parser.add_argument(
            "--output-dir",
            default="outputs",
            help="Directory to save outputs"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        
        # Add project-specific arguments
        self._add_project_arguments(parser)
        
        return parser
    
    @abstractmethod
    def _add_project_arguments(self, parser: argparse.ArgumentParser):
        """
        Add project-specific arguments to the parser.
        
        Args:
            parser: ArgumentParser instance to add arguments to
        """
        pass
    
    @abstractmethod
    def run_command(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Execute the command with given arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Dict containing execution results
        """
        pass
    
    def main(self):
        """Main entry point for the CLI."""
        args = self.parser.parse_args()
        
        if args.verbose:
            import logging
            logging.basicConfig(level=logging.DEBUG)
        
        try:
            results = self.run_command(args)
            print(f"✅ {self.project_name} completed successfully!")
            return results
        except Exception as e:
            print(f"❌ {self.project_name} failed: {str(e)}")
            raise