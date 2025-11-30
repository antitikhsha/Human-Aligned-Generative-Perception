#!/bin/bash

# Phase 5: Closed-Loop Human Feedback - AWS Deployment Script
# This script deploys the complete infrastructure for human feedback collection

set -e  # Exit on any error

# Configuration
PROJECT_NAME="VLR-Phase5"
AWS_REGION="us-east-2"
AWS_PROFILE="VLR_project"
STACK_NAME="${PROJECT_NAME}-Infrastructure"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check if AWS CLI is configured
check_aws_config() {
    print_status "Checking AWS configuration..."
    
    if ! aws sts get-caller-identity --profile $AWS_PROFILE >/dev/null 2>&1; then
        print_error "AWS CLI not configured for profile $AWS_PROFILE"
        print_error "Please run: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    
    print_status "AWS configuration verified for profile: $AWS_PROFILE"
}

# Function to validate prerequisites
validate_prerequisites() {
    print_status "Validating prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check required files
    required_files=(
        "phase5_cloudformation_template.yaml"
        "config/phase5_config.yaml"
        "step5_1_image_generation_worker.py"
        "step5_2_feedback_collection_lambda.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file not found: $file"
            exit 1
        fi
    done
    
    print_status "Prerequisites validated successfully"
}

# Function to create S3 bucket for deployment artifacts
create_deployment_bucket() {
    local bucket_name="$PROJECT_NAME-deployment-artifacts-$(date +%s)"
    bucket_name=$(echo $bucket_name | tr '[:upper:]' '[:lower:]')
    
    print_status "Creating deployment artifacts bucket: $bucket_name"
    
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3 mb s3://$bucket_name --profile $AWS_PROFILE
    else
        aws s3 mb s3://$bucket_name --region $AWS_REGION --profile $AWS_PROFILE
    fi
    
    echo $bucket_name
}

# Function to package and upload Lambda functions
package_lambda_functions() {
    local deployment_bucket=$1
    
    print_status "Packaging Lambda functions..."
    
    # Create temporary directory for packaging
    mkdir -p /tmp/lambda-packages
    
    # Package feedback collection functions
    cd /tmp/lambda-packages
    cp $OLDPWD/step5_2_feedback_collection_lambda.py ./index.py
    
    # Install dependencies if requirements exist
    if [ -f "$OLDPWD/requirements.txt" ]; then
        pip install -r $OLDPWD/requirements.txt -t .
    fi
    
    zip -r feedback-functions.zip .
    
    # Upload to S3
    aws s3 cp feedback-functions.zip s3://$deployment_bucket/lambda/ --profile $AWS_PROFILE
    
    cd $OLDPWD
    rm -rf /tmp/lambda-packages
    
    print_status "Lambda functions packaged and uploaded"
}

# Function to create CloudFormation parameters
create_cf_parameters() {
    local model_bucket=$1
    local key_pair_name=$2
    
    cat > cf_parameters.json <<EOF
[
    {
        "ParameterKey": "ProjectName",
        "ParameterValue": "$PROJECT_NAME"
    },
    {
        "ParameterKey": "KeyPairName",
        "ParameterValue": "$key_pair_name"
    },
    {
        "ParameterKey": "ModelBucket",
        "ParameterValue": "$model_bucket"
    },
    {
        "ParameterKey": "InstanceType",
        "ParameterValue": "g4dn.xlarge"
    }
]
EOF
}

# Function to deploy CloudFormation stack
deploy_cloudformation() {
    local deployment_bucket=$1
    local model_bucket=$2
    
    print_status "Deploying CloudFormation stack: $STACK_NAME"
    
    # Get or create key pair
    local key_pair_name="${PROJECT_NAME}-keypair"
    if ! aws ec2 describe-key-pairs --key-names $key_pair_name --profile $AWS_PROFILE >/dev/null 2>&1; then
        print_status "Creating new key pair: $key_pair_name"
        aws ec2 create-key-pair --key-name $key_pair_name --profile $AWS_PROFILE --query 'KeyMaterial' --output text > ${key_pair_name}.pem
        chmod 400 ${key_pair_name}.pem
        print_status "Key pair saved to: ${key_pair_name}.pem"
    fi
    
    # Create parameters file
    create_cf_parameters $model_bucket $key_pair_name
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name $STACK_NAME --profile $AWS_PROFILE >/dev/null 2>&1; then
        print_status "Updating existing CloudFormation stack..."
        aws cloudformation update-stack \
            --stack-name $STACK_NAME \
            --template-body file://phase5_cloudformation_template.yaml \
            --parameters file://cf_parameters.json \
            --capabilities CAPABILITY_NAMED_IAM \
            --profile $AWS_PROFILE
            
        aws cloudformation wait stack-update-complete \
            --stack-name $STACK_NAME \
            --profile $AWS_PROFILE
    else
        print_status "Creating new CloudFormation stack..."
        aws cloudformation create-stack \
            --stack-name $STACK_NAME \
            --template-body file://phase5_cloudformation_template.yaml \
            --parameters file://cf_parameters.json \
            --capabilities CAPABILITY_NAMED_IAM \
            --profile $AWS_PROFILE
            
        aws cloudformation wait stack-create-complete \
            --stack-name $STACK_NAME \
            --profile $AWS_PROFILE
    fi
    
    print_status "CloudFormation stack deployed successfully"
    
    # Clean up parameters file
    rm -f cf_parameters.json
}

# Function to update Lambda function codes
update_lambda_functions() {
    print_status "Updating Lambda function codes..."
    
    # Get function names from CloudFormation outputs
    local functions=(
        "feedback-interface"
        "get-triplet"
        "submit-response"
        "skip-triplet"
        "image-processing"
    )
    
    for func in "${functions[@]}"; do
        local func_name="${PROJECT_NAME}-${func}"
        
        print_status "Updating function: $func_name"
        
        # Create zip file for this function
        cd /tmp
        rm -rf lambda-update
        mkdir lambda-update
        cd lambda-update
        
        # Copy the appropriate code
        cp $OLDPWD/step5_2_feedback_collection_lambda.py ./index.py
        
        # Create zip
        zip -r function.zip .
        
        # Update function
        aws lambda update-function-code \
            --function-name $func_name \
            --zip-file fileb://function.zip \
            --profile $AWS_PROFILE || true  # Continue if function doesn't exist
        
        cd $OLDPWD
        rm -rf /tmp/lambda-update
    done
    
    print_status "Lambda functions updated"
}

# Function to upload worker scripts to S3
upload_worker_scripts() {
    local model_bucket=$1
    
    print_status "Uploading worker scripts to model bucket: $model_bucket"
    
    # Upload generation worker script
    aws s3 cp step5_1_image_generation_worker.py s3://$model_bucket/scripts/ --profile $AWS_PROFILE
    
    # Upload configuration
    aws s3 cp config/phase5_config.yaml s3://$model_bucket/config/ --profile $AWS_PROFILE
    
    print_status "Worker scripts uploaded successfully"
}

# Function to launch generation instances
launch_generation_instances() {
    print_status "Launching image generation instances..."
    
    # Get launch template from CloudFormation outputs
    local template_id=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs[?OutputKey==`LaunchTemplateId`].OutputValue' \
        --output text \
        --profile $AWS_PROFILE)
    
    if [ "$template_id" != "None" ] && [ ! -z "$template_id" ]; then
        # Launch instances using the launch template
        aws ec2 run-instances \
            --launch-template LaunchTemplateId=$template_id \
            --min-count 1 \
            --max-count 2 \
            --profile $AWS_PROFILE
            
        print_status "Generation instances launched with template: $template_id"
    else
        print_warning "Launch template not found, skipping instance launch"
    fi
}

# Function to display deployment information
display_deployment_info() {
    print_header "=== DEPLOYMENT COMPLETED SUCCESSFULLY ==="
    
    # Get CloudFormation outputs
    local outputs=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs' \
        --profile $AWS_PROFILE)
    
    echo "CloudFormation Stack: $STACK_NAME"
    echo "AWS Region: $AWS_REGION"
    echo
    
    # Parse and display key outputs
    echo "Key Resources:"
    echo $outputs | jq -r '.[] | "  - \(.OutputKey): \(.OutputValue)"' 2>/dev/null || echo "  (Install jq for formatted output)"
    
    echo
    print_status "Web Interface URL:"
    local web_url=$(echo $outputs | jq -r '.[] | select(.OutputKey=="FeedbackWebInterface") | .OutputValue' 2>/dev/null)
    if [ ! -z "$web_url" ] && [ "$web_url" != "null" ]; then
        echo "  $web_url"
    else
        print_warning "Web interface URL not found in outputs"
    fi
    
    echo
    print_status "Next Steps:"
    echo "  1. Upload your trained StyleGAN2 model to the model bucket"
    echo "  2. Start image generation tasks using the orchestrator script"
    echo "  3. Share the web interface URL with human participants"
    echo "  4. Monitor costs and usage in AWS console"
    
    print_warning "Remember to clean up resources when done to avoid charges!"
}

# Function to setup cost monitoring
setup_cost_monitoring() {
    print_status "Setting up cost monitoring..."
    
    # Create billing alarm
    aws cloudwatch put-metric-alarm \
        --alarm-name "${PROJECT_NAME}-HighCostAlert" \
        --alarm-description "Alert when estimated charges exceed threshold" \
        --metric-name EstimatedCharges \
        --namespace AWS/Billing \
        --statistic Maximum \
        --period 86400 \
        --threshold 100 \
        --comparison-operator GreaterThanThreshold \
        --evaluation-periods 1 \
        --unit USD \
        --dimensions Name=Currency,Value=USD \
        --profile $AWS_PROFILE || true
    
    print_status "Cost monitoring configured"
}

# Function to run tests
run_basic_tests() {
    print_status "Running basic tests..."
    
    # Test API Gateway endpoint
    local api_url=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs[?OutputKey==`FeedbackAPIEndpoint`].OutputValue' \
        --output text \
        --profile $AWS_PROFILE)
    
    if [ ! -z "$api_url" ] && [ "$api_url" != "None" ]; then
        print_status "Testing API endpoint: $api_url"
        if curl -s -f "$api_url/feedback" >/dev/null; then
            print_status "API endpoint is responding"
        else
            print_warning "API endpoint test failed - may need time to initialize"
        fi
    fi
    
    print_status "Basic tests completed"
}

# Main deployment function
main() {
    print_header "=== Phase 5: Closed-Loop Human Feedback Deployment ==="
    print_status "Starting deployment for $PROJECT_NAME"
    
    # Validate prerequisites
    validate_prerequisites
    check_aws_config
    
    # Get model bucket from user input or use default
    if [ -z "$1" ]; then
        print_error "Usage: $0 <model-bucket-name>"
        print_error "Please provide the S3 bucket name containing your trained models"
        exit 1
    fi
    
    local model_bucket=$1
    
    # Verify model bucket exists
    if ! aws s3 ls s3://$model_bucket --profile $AWS_PROFILE >/dev/null 2>&1; then
        print_error "Model bucket does not exist or is not accessible: $model_bucket"
        exit 1
    fi
    
    print_status "Using model bucket: $model_bucket"
    
    # Create deployment bucket
    local deployment_bucket=$(create_deployment_bucket)
    
    # Package and upload Lambda functions
    package_lambda_functions $deployment_bucket
    
    # Deploy CloudFormation stack
    deploy_cloudformation $deployment_bucket $model_bucket
    
    # Update Lambda function codes
    update_lambda_functions
    
    # Upload worker scripts
    upload_worker_scripts $model_bucket
    
    # Setup cost monitoring
    setup_cost_monitoring
    
    # Launch generation instances (optional)
    if [ "$2" = "--launch-instances" ]; then
        launch_generation_instances
    fi
    
    # Run basic tests
    run_basic_tests
    
    # Display deployment information
    display_deployment_info
    
    print_status "Deployment completed successfully!"
}

# Handle script termination
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f cf_parameters.json
    rm -f ${PROJECT_NAME}-keypair.pem 2>/dev/null || true
}

trap cleanup EXIT

# Run main function with all arguments
main "$@"
