#!/bin/bash

# AWS StyleGAN2 Training Instance Management Script
# Handles instance lifecycle, monitoring, and cost optimization

set -e

# Configuration
PROJECT_NAME="VLR-StyleGAN2"
REGION="us-east-2"
STACK_NAME="${PROJECT_NAME}-infrastructure"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}[HEADER]${NC} $1"; }

# Function to get stack outputs
get_stack_output() {
    local output_key=$1
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='$output_key'].OutputValue" \
        --output text
}

# Function to check if AWS CLI is configured
check_aws_config() {
    if ! aws sts get-caller-identity &>/dev/null; then
        print_error "AWS CLI not configured"
        exit 1
    fi
}

# Function to get training instances
get_training_instances() {
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters \
            "Name=tag:Project,Values=$PROJECT_NAME" \
            "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query "Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,Tags[?Key=='Name'].Value|[0],PublicIpAddress,LaunchTime]" \
        --output table
}

# Function to launch training instance with optimized configuration
launch_optimized_instance() {
    local instance_type=${1:-"g4dn.xlarge"}
    local spot_instances=${2:-"false"}
    local auto_shutdown=${3:-"true"}
    local training_duration=${4:-"24"} # hours
    
    print_header "Launching optimized training instance"
    
    # Get infrastructure components
    local launch_template=$(get_stack_output "LaunchTemplateId")
    local subnet_id=$(get_stack_output "SubnetId")
    local security_group=$(get_stack_output "SecurityGroupId")
    
    if [[ -z "$launch_template" || -z "$subnet_id" ]]; then
        print_error "Infrastructure not found. Please deploy infrastructure first."
        exit 1
    fi
    
    local instance_name="stylegan2-training-$(date +%Y%m%d-%H%M%S)"
    
    # Create user data script for auto-setup
    local user_data=$(cat << 'EOF'
#!/bin/bash

# Update system
apt-get update

# Create auto-shutdown timer if enabled
if [[ "$AUTO_SHUTDOWN" == "true" ]]; then
    # Schedule shutdown after training duration
    echo "sudo shutdown -h +$((TRAINING_DURATION * 60))" | at now
    echo "Auto-shutdown scheduled for $TRAINING_DURATION hours from now"
fi

# Setup training environment
cd /home/ubuntu
sudo -u ubuntu bash << 'USEREOF'

# Activate conda environment
source /home/ubuntu/anaconda3/bin/activate pytorch

# Clone StyleGAN2 repository (replace with your actual repo)
git clone https://github.com/your-username/stylegan2-things.git stylegan2
cd stylegan2

# Install requirements
pip install -r requirements.txt

# Download training scripts from S3 (if stored there)
aws s3 cp s3://your-scripts-bucket/stylegan2_aws_trainer.py . || echo "Training script not in S3, using local"
aws s3 cp s3://your-scripts-bucket/training_config.yaml . || echo "Config not in S3, using local"

# Create training start script
cat > start_training.sh << 'TRAINEOF'
#!/bin/bash

# Activate environment
source /home/ubuntu/anaconda3/bin/activate pytorch

# Start training with monitoring
python stylegan2_aws_trainer.py \
    --config training_config.yaml \
    --distributed 2>&1 | tee training.log

# Upload final logs and results
aws s3 sync runs/ s3://$(aws cloudformation describe-stacks --stack-name VLR-StyleGAN2-infrastructure --region us-east-2 --query "Stacks[0].Outputs[?OutputKey=='OutputBucketName'].OutputValue" --output text)/runs/
aws s3 cp training.log s3://$(aws cloudformation describe-stacks --stack-name VLR-StyleGAN2-infrastructure --region us-east-2 --query "Stacks[0].Outputs[?OutputKey=='OutputBucketName'].OutputValue" --output text)/logs/

# Send completion notification (if configured)
if [[ -n "$SLACK_WEBHOOK" ]]; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"StyleGAN2 training completed on instance '$HOSTNAME'"}' \
        $SLACK_WEBHOOK
fi

echo "Training completed. Instance will auto-shutdown in 10 minutes if auto-shutdown is enabled."
TRAINEOF

chmod +x start_training.sh

# Setup Jupyter notebook for monitoring (optional)
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py

USEREOF

# Signal CloudFormation
/opt/aws/bin/cfn-signal -e $? --stack $STACK_NAME --resource TrainingInstance --region $REGION

EOF
)

    # Encode user data
    local encoded_user_data=$(echo "$user_data" | base64 -w 0)
    
    # Launch instance
    local launch_params=""
    
    if [[ "$spot_instances" == "true" ]]; then
        # Use spot instances for cost savings
        print_info "Launching spot instance (cheaper but can be interrupted)"
        
        # Create spot instance request
        local spot_price="0.50" # Adjust based on instance type
        
        local instance_id=$(aws ec2 request-spot-instances \
            --region "$REGION" \
            --spot-price "$spot_price" \
            --instance-count 1 \
            --type "one-time" \
            --launch-specification "{
                \"ImageId\": \"ami-0c02fb55956c7d316\",
                \"InstanceType\": \"$instance_type\",
                \"KeyName\": \"$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query "Stacks[0].Parameters[?ParameterKey=='KeyPairName'].ParameterValue" --output text)\",
                \"SecurityGroups\": [\"$security_group\"],
                \"SubnetId\": \"$subnet_id\",
                \"UserData\": \"$encoded_user_data\",
                \"IamInstanceProfile\": {
                    \"Name\": \"${PROJECT_NAME}-instance-profile\"
                }
            }" \
            --query "SpotInstanceRequests[0].SpotInstanceRequestId" \
            --output text)
        
        print_info "Spot instance request created: $instance_id"
        
    else
        # Use on-demand instances
        print_info "Launching on-demand instance"
        
        local instance_id=$(aws ec2 run-instances \
            --region "$REGION" \
            --launch-template LaunchTemplateId="$launch_template" \
            --subnet-id "$subnet_id" \
            --user-data "$encoded_user_data" \
            --tag-specifications "ResourceType=instance,Tags=[
                {Key=Name,Value=$instance_name},
                {Key=Project,Value=$PROJECT_NAME},
                {Key=Purpose,Value=StyleGAN2Training},
                {Key=AutoShutdown,Value=$auto_shutdown},
                {Key=TrainingDuration,Value=${training_duration}h}
            ]" \
            --query "Instances[0].InstanceId" \
            --output text)
    fi
    
    print_info "Instance launched: $instance_id"
    print_info "Waiting for instance to be running..."
    
    # Wait for instance to be running
    aws ec2 wait instance-running \
        --instance-ids "$instance_id" \
        --region "$REGION"
    
    # Get instance details
    local instance_details=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query "Reservations[0].Instances[0].[PublicIpAddress,PrivateIpAddress,InstanceType]" \
        --output text)
    
    local public_ip=$(echo "$instance_details" | cut -f1)
    local private_ip=$(echo "$instance_details" | cut -f2)
    local actual_type=$(echo "$instance_details" | cut -f3)
    
    print_info "Instance is running!"
    print_info "Instance ID: $instance_id"
    print_info "Instance Type: $actual_type"
    print_info "Public IP: $public_ip"
    print_info "Private IP: $private_ip"
    
    # Create connection info file
    cat > "instance_${instance_id}_info.txt" << EOF
# StyleGAN2 Training Instance Information
Instance ID: $instance_id
Instance Name: $instance_name
Instance Type: $actual_type
Public IP: $public_ip
Private IP: $private_ip
Launch Time: $(date)
Auto Shutdown: $auto_shutdown
Training Duration: ${training_duration}h

# Connection Commands
SSH: ssh -i your-key.pem ubuntu@$public_ip
Jupyter: http://$public_ip:8888
TensorBoard: http://$public_ip:6006

# Training Commands
Start Training: ./start_training.sh
Monitor Progress: tail -f training.log
Check GPU: nvidia-smi

# Cost Monitoring
Estimated Cost/Hour: \$$(get_instance_cost $actual_type)
Estimated Total Cost: \$$(echo "scale=2; $(get_instance_cost $actual_type) * $training_duration" | bc)
EOF
    
    print_info "Instance information saved to instance_${instance_id}_info.txt"
    
    # Setup monitoring
    setup_monitoring "$instance_id"
    
    return 0
}

# Function to get instance hourly cost
get_instance_cost() {
    local instance_type=$1
    
    case $instance_type in
        "g4dn.xlarge") echo "0.526" ;;
        "g4dn.2xlarge") echo "0.752" ;;
        "p3.2xlarge") echo "3.06" ;;
        "p3.8xlarge") echo "12.24" ;;
        *) echo "0.50" ;;
    esac
}

# Function to setup CloudWatch monitoring
setup_monitoring() {
    local instance_id=$1
    
    print_info "Setting up CloudWatch monitoring for $instance_id"
    
    # Create custom metrics alarm for GPU utilization (requires CloudWatch agent)
    aws cloudwatch put-metric-alarm \
        --region "$REGION" \
        --alarm-name "${PROJECT_NAME}-gpu-utilization-${instance_id}" \
        --alarm-description "GPU utilization for StyleGAN2 training" \
        --metric-name "GPUUtilization" \
        --namespace "CWAgent" \
        --statistic "Average" \
        --period 300 \
        --evaluation-periods 3 \
        --threshold 10 \
        --comparison-operator "LessThanThreshold" \
        --dimensions "Name=InstanceId,Value=$instance_id" || true
    
    # Create billing alarm
    aws cloudwatch put-metric-alarm \
        --region "$REGION" \
        --alarm-name "${PROJECT_NAME}-billing-${instance_id}" \
        --alarm-description "Billing alarm for training instance" \
        --metric-name "EstimatedCharges" \
        --namespace "AWS/Billing" \
        --statistic "Maximum" \
        --period 86400 \
        --evaluation-periods 1 \
        --threshold 50 \
        --comparison-operator "GreaterThanThreshold" \
        --dimensions "Name=Currency,Value=USD" || true
}

# Function to monitor training progress
monitor_training() {
    local instance_id=$1
    
    if [[ -z "$instance_id" ]]; then
        print_error "Instance ID required"
        exit 1
    fi
    
    # Get instance IP
    local public_ip=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --region "$REGION" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text)
    
    if [[ "$public_ip" == "None" || -z "$public_ip" ]]; then
        print_error "Instance not running or no public IP"
        exit 1
    fi
    
    print_header "Training Progress Monitor for $instance_id"
    print_info "Public IP: $public_ip"
    print_info "SSH: ssh -i your-key.pem ubuntu@$public_ip"
    print_info "Jupyter: http://$public_ip:8888"
    print_info "TensorBoard: http://$public_ip:6006"
    
    # Show recent CloudWatch metrics
    print_info "Recent CPU utilization:"
    aws cloudwatch get-metric-statistics \
        --region "$REGION" \
        --namespace "AWS/EC2" \
        --metric-name "CPUUtilization" \
        --dimensions "Name=InstanceId,Value=$instance_id" \
        --start-time "$(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%S)" \
        --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        --period 300 \
        --statistics "Average" \
        --query "Datapoints[*].[Timestamp,Average]" \
        --output table || print_warn "No CPU metrics available yet"
}

# Function to stop and terminate instances
cleanup_instances() {
    local action=${1:-"stop"} # stop or terminate
    
    print_header "Instance cleanup: $action"
    
    # Get running instances
    local instances=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters \
            "Name=tag:Project,Values=$PROJECT_NAME" \
            "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[*].Instances[*].InstanceId" \
        --output text)
    
    if [[ -z "$instances" ]]; then
        print_info "No running instances found"
        return 0
    fi
    
    print_info "Found instances: $instances"
    
    if [[ "$action" == "terminate" ]]; then
        print_warn "This will PERMANENTLY delete the instances and all local data!"
        read -p "Are you sure? (yes/no): " confirmation
        if [[ "$confirmation" != "yes" ]]; then
            print_info "Termination cancelled"
            return 0
        fi
        
        aws ec2 terminate-instances \
            --instance-ids $instances \
            --region "$REGION"
        print_info "Instances termination initiated"
        
    else
        aws ec2 stop-instances \
            --instance-ids $instances \
            --region "$REGION"
        print_info "Instances stop initiated"
    fi
}

# Function to estimate training costs
estimate_costs() {
    local instance_type=${1:-"g4dn.xlarge"}
    local training_hours=${2:-"24"}
    local spot_discount=${3:-"0"} # percentage
    
    print_header "Cost Estimation"
    
    local hourly_cost=$(get_instance_cost "$instance_type")
    local base_cost=$(echo "scale=2; $hourly_cost * $training_hours" | bc)
    
    if [[ "$spot_discount" -gt 0 ]]; then
        local spot_cost=$(echo "scale=2; $base_cost * (100 - $spot_discount) / 100" | bc)
        print_info "On-demand cost: \$${base_cost} ($training_hours hours @ \$${hourly_cost}/hour)"
        print_info "Spot cost (${spot_discount}% discount): \$${spot_cost}"
    else
        print_info "Estimated cost: \$${base_cost} ($training_hours hours @ \$${hourly_cost}/hour)"
    fi
    
    # Show additional costs
    print_info "Additional costs to consider:"
    print_info "- S3 storage: ~\$0.023/GB/month"
    print_info "- Data transfer: \$0.09/GB outbound"
    print_info "- CloudWatch logs: \$0.50/GB ingested"
}

# Function to show help
show_help() {
    cat << EOF
AWS StyleGAN2 Training Instance Management

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  launch [TYPE] [SPOT] [AUTO_SHUTDOWN] [DURATION]
                        Launch optimized training instance
                        TYPE: g4dn.xlarge, g4dn.2xlarge, p3.2xlarge, p3.8xlarge
                        SPOT: true/false (default: false)
                        AUTO_SHUTDOWN: true/false (default: true)
                        DURATION: hours (default: 24)
  
  list                  List all training instances
  
  monitor INSTANCE_ID   Monitor training progress
  
  stop                  Stop all running instances
  
  terminate             Terminate all instances (PERMANENT)
  
  estimate TYPE HOURS   Estimate training costs
  
  help                  Show this help

Examples:
  $0 launch g4dn.xlarge true true 12    # Launch spot instance with 12h auto-shutdown
  $0 launch p3.2xlarge false false      # Launch on-demand instance without auto-shutdown
  $0 monitor i-1234567890abcdef0        # Monitor specific instance
  $0 estimate p3.8xlarge 48             # Estimate cost for 48h training
  $0 stop                               # Stop all running instances

EOF
}

# Main script logic
case "$1" in
    "launch")
        check_aws_config
        launch_optimized_instance "$2" "$3" "$4" "$5"
        ;;
    "list")
        check_aws_config
        print_header "Training Instances"
        get_training_instances
        ;;
    "monitor")
        check_aws_config
        monitor_training "$2"
        ;;
    "stop")
        check_aws_config
        cleanup_instances "stop"
        ;;
    "terminate")
        check_aws_config
        cleanup_instances "terminate"
        ;;
    "estimate")
        estimate_costs "$2" "$3" "$4"
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        print_error "Invalid command: $1"
        show_help
        exit 1
        ;;
esac
