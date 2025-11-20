// Corrected Jenkinsfile
pipeline {
    agent any // Should be 'agent { docker ... }' or 'agent { node ... }' with GPU access for training

    environment {
        // 🚨 CORRECTION 1: Define the final, publicly visible Docker Hub repository name here
        // The image name for the Financial Sentiment API service
        DOCKER_REPO_NAME = 'nikhilesh611/financial-sentiment-api' 
        // Local name for the training container (doesn't need to be on Docker Hub)
        TRAINING_IMAGE_NAME = 'financial-sentiment-trainer'
        DOCKER_HUB_CREDENTIALS_ID = 'dockerhub-creds'
    }
    
    stages {
        stage('Checkout Code') {
            steps {
                git url: 'https://github.com/Nikhilesh611/fin-mlops', branch: 'main' 
            }
        }

        stage('Build Training Image') {
            steps {
                script {
                    // Use the defined local name
                    sh "docker build -t ${TRAINING_IMAGE_NAME}:\${GIT_COMMIT} ."
                }
            }
        }

        stage('Run Fine-Tuning and Model Saving') {
            steps {
                // Run the fine-tuning script inside the container using the local name
                sh """
                docker run --rm \
                    -v \$(pwd)/model_artifacts:/app/model_artifacts \
                    ${TRAINING_IMAGE_NAME}:\${GIT_COMMIT} \
                    python train.py
                """
            }
        }

        stage('Model Validation and Testing') {
            steps {
                sh 'echo "Model validation complete (placeholder: run validation script here)"'
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'model_artifacts/**/*', fingerprint: true
            }
        }

        stage('Build Deployment Image') {
            steps {
                script {
                    // 🚨 CORRECTION 2: Use the final DOCKER_REPO_NAME as the local tag here
                    // This uses the official name but tags it locally first
                    sh "docker build -t ${DOCKER_REPO_NAME}:\${GIT_COMMIT} ."
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                script {
                    // 🚨 CORRECTION 3: Use DOCKER_REPO_NAME consistently
                    def commitId = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    
                    withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS_ID}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                        // 1. Explicitly log in
                        sh "echo \$DOCKER_PASSWORD | docker login -u \$DOCKER_USERNAME --password-stdin"
                        
                        // 2. Push the two tags (using the repository name)
                        sh "docker push ${DOCKER_REPO_NAME}:${commitId}"
                        sh "docker push ${DOCKER_REPO_NAME}:latest" 
                        
                        // 3. Logout for security
                        sh "docker logout" 
                    }
                }
            }
        }
    }
}