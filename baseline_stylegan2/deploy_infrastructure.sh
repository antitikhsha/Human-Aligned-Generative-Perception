#!/bin/bash

# AWS StyleGAN2 Infrastructure Deployment Script
# Usage: ./deploy_infrastructure.sh [create|update|delete] [key-pair-name] [instance-type]

set -e

# Configuration
PROJECT_NAME="VLR-StyleGAN2"
STACK_NAME="${PROJECT_NAME}-infrastructure"
TEMPLATE_FILE="stylegan2_aws_infrastructure.yaml"
REGION="us-east-2"

# Function to print colored output
print_info() {
    echo -e "\e[32m[INFO]\e[0m $1"
}

print_error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

print_warning() {
    echo -e "\e[33m[WARNING]\e[0m $1"
}

# Function to check if AWS CLI is configured
check_aws_config() {
    if ! aws sts get-caller-identity &>/dev/null; then
        print_error "AWS CLI not configured or credentials invalid"
        print_info "Please run: aws configure"
        exit 1
    fi
    print_info "AWS CLI configured for account: $(aws sts get-caller-identity --query Account --output text)"
}

# Function to check if key pair exists
check_key_pair() {
    local key_name=$1
    if ! aws ec2 describe-key-pairs --key-names "$key_name" --region "$REGION" &>/dev/null; then
        print_error "Key pair '$key_name' not found in region $REGION"
        print_info "Create a key pair with: aws ec2 create-key-pair --key-name $key_name --region $REGION"
        exit 1
    fi
    print_info "Key pair '$key_name' found"
}

# Function to create stack
create_stack() {
    local key_pair=$1
    local instance_type=${2:-g4dn.xlarge}
    
    print_info "Creating CloudFormation stack: $STACK_NAME"
    
    aws cloudformation create-stack \
        --stack-name "$STACK_NAME" \
        --template-body "file://$TEMPLATE_FILE" \
        --capabilities CAPABILITY_NAMED_IAM \
        --parameters \
            ParameterKey=ProjectName,ParameterValue="$PROJECT_NAME" \
            ParameterKey=KeyPairName,ParameterValue="$key_pair" \
            ParameterKey=InstanceType,ParameterValue="$instance_type" \
        --region "$REGION" \
        --tags \
            Key=Project,Value="$PROJECT_NAME" \
            Key=Environment,Value=Development \
            Key=Purpose,Value=StyleGAN2Training
    
    print_info "Stack creation initiated. Waiting for completion..."
    
    aws cloudformation wait stack-create-complete \
        --stack-name "$STACK_NAME" \
        --region "$REGION"
    
    if [ $? -eq 0 ]; then
        print_info "Stack created successfully!"
        show_outputs
    else
        print_error "Stack creation failed"
        exit 1
    fi
}

# Function to update stack
update_stack() {
    local key_pair=$1
    local instance_type=${2:-g4dn.xlarge}
    
    print_info "Updating CloudFormation stack: $STACK_NAME"
    
    aws cloudformation update-stack \
        --stack-name "$STACK_NAME" \
        --template-body "file://$TEMPLATE_FILE" \
        --capabilities CAPABILITY_NAMED_IAM \
        --parameters \
            ParameterKey=ProjectName,ParameterValue="$PROJECT_NAME" \
            ParameterKey=KeyPairName,ParameterValue="$key_pair" \
            ParameterKey=InstanceType,ParameterValue="$instance_type" \
        --region "$REGION" \
        --tags \
            Key=Project,Value="$PROJECT_NAME" \
            Key=Environment,Value=Development \
            Key=Purpose,Value=StyleGAN2Training
    
    print_info "Stack update initiated. Waiting for completion..."
    
    aws cloudformation wait stack-update-complete \
        --stack-name "$STACK_NAME" \
        --region "$REGION"
    
    if [ $? -eq 0 ]; then
        print_info "Stack updated successfully!"
        show_outputs
    else
        print_error "Stack update failed"
        exit 1
    fi
}

# Function to delete stack
delete_stack() {
    print_warning "This will delete all infrastructure including S3 buckets and data!"
    read -p "Are you sure you want to delete the stack? (yes/no): " confirmation
    
    if [[ $confirmation == "yes" ]]; then
        # First, empty S3 buckets to allow stack deletion
        empty_s3_buckets
        
        print_info "Deleting CloudFormation stack: $STACK_NAME"
        
        aws cloudformation delete-stack \
            --stack-name "$STACK_NAME" \
            --region "$REGION"
        
        print_info "Stack deletion initiated. Waiting for completion..."
        
        aws cloudformation wait stack-delete-complete \
            --stack-name "$STACK_NAME" \
            --region "$REGION"
        
        if [ $? -eq 0 ]; then
            print_info "Stack deleted successfully!"
        else
            print_error "Stack deletion failed"
            exit 1
        fi
    else
        print_info "Stack deletion cancelled"
    fi
}

# Function to empty S3 buckets
empty_s3_buckets() {
    print_info "Emptying S3 buckets..."
    
    # Get bucket names from stack outputs
    local data_bucket=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
        --output text)
    
    local checkpoint_bucket=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='CheckpointBucketName'].OutputValue" \
        --output text)
    
    local output_bucket=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='OutputBucketName'].OutputValue" \
        --output text)
    
    # Empty buckets
    for bucket in "$data_bucket" "$checkpoint_bucket" "$output_bucket"; do
        if [[ -n "$bucket" ]]; then
            print_info "Emptying bucket: $bucket"
            aws s3 rm "s3://$bucket" --recursive --region "$REGION" || true
        fi
    done
}

# Function to show stack outputs
show_outputs() {
    print_info "Stack outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[*].[OutputKey,OutputValue]" \
        --output table
}

# Function to create and start training instance
launch_training_instance() {
    local instance_name=${1:-"stylegan2-training-$(date +%Y%m%d-%H%M%S)"}
    
    print_info "Launching training instance: $instance_name"
    
    # Get launch template ID from stack outputs
    local launch_template=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateId'].OutputValue" \
        --output text)
    
    # Get subnet ID from stack outputs
    local subnet_id=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='SubnetId'].OutputValue" \
        --output text)
    
    # Launch instance
    local instance_id=$(aws ec2 run-instances \
        --launch-template LaunchTemplateId="$launch_template" \
        --subnet-id "$subnet_id" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$instance_name},{Key=Project,Value=$PROJECT_NAME}]" \
        --region "$REGION" \
        --query "Instances[0].InstanceId" \
        --output text)
    
    print_info "Instance launched with ID: $instance_id"
    print_info "Waiting for instance to be running..."
    
    aws ec2 wait instance-running \
        --instance-ids "$instance_id" \
        --region "$REGION"
    
    # Get public IP
    local public_ip=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text)
    
    print_info "Instance is running!"
    print_info "Public IP: $public_ip"
    print_info "SSH command: ssh -i your-key.pem ubuntu@$public_ip"
    print_info "Setup command: ./setup_environment.sh"
    
    # Save instance info
    cat > instance_info.txt << EOF
Instance ID: $instance_id
Instance Name: $instance_name
Public IP: $public_ip
SSH Command: ssh -i your-key.pem ubuntu@$public_ip
Launch Time: $(date)
EOF
    
    print_info "Instance information saved to instance_info.txt"
}

# Function to show help
show_help() {
    cat << EOF
AWS StyleGAN2 Infrastructure Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  create      Create new infrastructure stack
  update      Update existing infrastructure stack
  delete      Delete infrastructure stack
  launch      Launch a new training instance
  outputs     Show stack outputs
  help        Show this help message

Options for create/update:
  KEY_PAIR_NAME    EC2 Key Pair name (required)
  INSTANCE_TYPE    EC2 instance type (optional, default: g4dn.xlarge)
                   Allowed: g4dn.xlarge, g4dn.2xlarge, p3.2xlarge, p3.8xlarge

Examples:
  $0 create my-key-pair
  $0 create my-key-pair p3.2xlarge
  $0 update my-key-pair g4dn.2xlarge
  $0 launch my-training-instance
  $0 delete
  $0 outputs

Prerequisites:
  - AWS CLI configured with appropriate credentials
  - EC2 Key Pair created in the target region ($REGION)
  - CloudFormation template file ($TEMPLATE_FILE) in current directory

EOF
}

# Main script logic
case "$1" in
    "create")
        if [[ -z "$2" ]]; then
            print_error "Key pair name required for create command"
            show_help
            exit 1
        fi
        check_aws_config
        check_key_pair "$2"
        create_stack "$2" "$3"
        ;;
    "update")
        if [[ -z "$2" ]]; then
            print_error "Key pair name required for update command"
            show_help
            exit 1
        fi
        check_aws_config
        check_key_pair "$2"
        update_stack "$2" "$3"
        ;;
    "delete")
        check_aws_config
        delete_stack
        ;;
    "launch")
        check_aws_config
        launch_training_instance "$2"
        ;;
    "outputs")
        check_aws_config
        show_outputs
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Invalid command: $1"
        show_help
        exit 1
        ;;
esac
