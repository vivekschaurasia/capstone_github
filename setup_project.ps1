# Define the project structure
$structure = @{
    "medical_ai_capstone" = @{
        "data" = @{
            "raw" = @{}
            "processed" = @{}
            "embeddings" = @{}
        }
        "src" = @{
            "preprocessing" = @{
                "process_text.py" = "#!/usr/bin/env python`n"
                "process_images.py" = "#!/usr/bin/env python`n"
            }
            "rag" = @{
                "embed_docs.py" = "#!/usr/bin/env python`n"
                "retrieve.py" = "#!/usr/bin/env python`n"
            }
            "llava" = @{
                "finetune.py" = "#!/usr/bin/env python`n"
                "inference.py" = "#!/usr/bin/env python`n"
            }
            "api" = @{
                "main.py" = "#!/usr/bin/env python`n"
                "models.py" = "#!/usr/bin/env python`n"
            }
            "utils" = @{
                "config.py" = "#!/usr/bin/env python`n"
                "logging.py" = "#!/usr/bin/env python`n"
            }
        }
        "scripts" = @{
            "train_llava.sh" = "#!/bin/bash`n"
            "evaluate.py" = "#!/usr/bin/env python`n"
        }
        "requirements.txt" = ""
        "README.md" = "# Medical AI Capstone Project`n"
        ".gitignore" = "data/`n__pycache__/`n*.pyc`nmedical_ai_env/`n"
    }
}

# Function to create directories and files
function Create-Structure {
    param (
        [string]$BasePath,
        [hashtable]$Structure
    )
    foreach ($name in $Structure.Keys) {
        $path = Join-Path $BasePath $name
        if ($Structure[$name] -is [hashtable]) {
            # Directory
            New-Item -Path $path -ItemType Directory -Force | Out-Null
            Create-Structure -BasePath $path -Structure $Structure[$name]
        } else {
            # File
            Set-Content -Path $path -Value $Structure[$name] -Force
        }
    }
}

# Run the script
Create-Structure -BasePath "." -Structure $structure
Write-Host "Project structure created successfully!"