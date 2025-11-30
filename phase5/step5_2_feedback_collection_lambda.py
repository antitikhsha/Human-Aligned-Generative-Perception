#!/usr/bin/env python3
"""
Step 5.2: Collect New Human Triplets - AWS Lambda Functions
Web-based interface for collecting human perceptual similarity judgments.
"""

import json
import boto3
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
import base64
from decimal import Decimal

# Lambda function for serving the web interface
def serve_feedback_interface(event, context):
    """
    AWS Lambda function to serve the human feedback collection interface.
    """
    try:
        # Get query parameters
        query_params = event.get('queryStringParameters') or {}
        user_id = query_params.get('user_id', f"anonymous_{uuid.uuid4().hex[:8]}")
        
        # Generate HTML for the feedback interface
        html_content = generate_feedback_html(user_id)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': html_content
        }
        
    except Exception as e:
        logging.error(f"Error serving feedback interface: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def generate_feedback_html(user_id: str) -> str:
    """Generate HTML for the feedback collection interface."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VLR Project - Human Perceptual Feedback Collection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .triplet-container {
                display: flex;
                justify-content: space-around;
                margin: 30px 0;
                gap: 20px;
            }
            .image-box {
                text-align: center;
                cursor: pointer;
                padding: 15px;
                border: 3px solid #ddd;
                border-radius: 10px;
                transition: all 0.3s;
                flex: 1;
                max-width: 300px;
            }
            .image-box:hover {
                border-color: #007bff;
                transform: scale(1.05);
            }
            .image-box.selected {
                border-color: #28a745;
                background-color: #f8f9fa;
            }
            .image-box img {
                max-width: 100%;
                height: 200px;
                object-fit: cover;
                border-radius: 5px;
            }
            .question {
                text-align: center;
                font-size: 18px;
                margin: 20px 0;
                font-weight: bold;
                color: #333;
            }
            .controls {
                text-align: center;
                margin: 30px 0;
            }
            .btn {
                padding: 12px 30px;
                margin: 0 10px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            .btn-primary {
                background-color: #007bff;
                color: white;
            }
            .btn-primary:hover {
                background-color: #0056b3;
            }
            .btn-secondary {
                background-color: #6c757d;
                color: white;
            }
            .btn-secondary:hover {
                background-color: #545b62;
            }
            .progress {
                background-color: #e9ecef;
                border-radius: 5px;
                height: 20px;
                margin: 20px 0;
            }
            .progress-bar {
                background-color: #007bff;
                height: 100%;
                border-radius: 5px;
                transition: width 0.3s;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
            }
            .loading {
                text-align: center;
                padding: 50px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>VLR Project: Human Perceptual Feedback Collection</h1>
                <p>Help us improve AI image generation by providing your perceptual judgments!</p>
            </div>
            
            <div class="stats">
                <div>User ID: <strong id="user-id">{user_id}</strong></div>
                <div>Completed: <strong id="completed-count">0</strong></div>
                <div>Remaining: <strong id="remaining-count">-</strong></div>
            </div>
            
            <div class="progress">
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Loading image triplet...</p>
            </div>
            
            <div id="triplet-interface" style="display: none;">
                <div class="question">
                    Which image is the <strong>odd one out</strong>? Click on the image that seems most different from the other two.
                </div>
                
                <div class="triplet-container" id="triplet-container">
                    <!-- Images will be loaded here -->
                </div>
                
                <div class="controls">
                    <button class="btn btn-primary" id="submit-btn" onclick="submitResponse()" disabled>
                        Submit Response
                    </button>
                    <button class="btn btn-secondary" id="skip-btn" onclick="skipTriplet()">
                        Skip This Triplet
                    </button>
                </div>
            </div>
            
            <div id="completion-message" style="display: none; text-align: center;">
                <h2>Thank you for your participation!</h2>
                <p>You have completed the feedback collection session.</p>
                <p>Your responses will help improve human-aligned AI image generation.</p>
            </div>
        </div>

        <script>
            // Global variables
            let currentTriplet = null;
            let selectedImageIndex = null;
            let completedCount = 0;
            let totalTriplets = 50; // Default, will be updated
            let userId = '{user_id}';
            
            // API configuration
            const API_ENDPOINT = window.location.origin + '/api';
            
            // Initialize the interface
            document.addEventListener('DOMContentLoaded', function() {
                loadNextTriplet();
            });
            
            async function loadNextTriplet() {
                try {
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('triplet-interface').style.display = 'none';
                    
                    const response = await fetch(`${{API_ENDPOINT}}/get-triplet?user_id=${{userId}}`);
                    const data = await response.json();
                    
                    if (data.completed) {
                        showCompletionMessage();
                        return;
                    }
                    
                    currentTriplet = data.triplet;
                    totalTriplets = data.total_triplets;
                    
                    displayTriplet(currentTriplet);
                    updateProgress();
                    
                } catch (error) {
                    console.error('Error loading triplet:', error);
                    alert('Error loading images. Please refresh the page.');
                }
            }
            
            function displayTriplet(triplet) {
                const container = document.getElementById('triplet-container');
                container.innerHTML = '';
                
                triplet.images.forEach((imageUrl, index) => {
                    const imageBox = document.createElement('div');
                    imageBox.className = 'image-box';
                    imageBox.onclick = () => selectImage(index);
                    
                    const img = document.createElement('img');
                    img.src = imageUrl;
                    img.alt = `Image ${{index + 1}}`;
                    img.onerror = () => {
                        console.error(`Failed to load image: ${{imageUrl}}`);
                        img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSIyMDAiIGZpbGw9IiNmNWY1ZjUiLz48dGV4dCB4PSI1MCIgeT0iMTAwIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM2NjYiPkltYWdlIEVycm9yPC90ZXh0Pjwvc3ZnPg==';
                    };
                    
                    const label = document.createElement('p');
                    label.textContent = `Image ${{index + 1}}`;
                    
                    imageBox.appendChild(img);
                    imageBox.appendChild(label);
                    container.appendChild(imageBox);
                });
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('triplet-interface').style.display = 'block';
                selectedImageIndex = null;
                updateSubmitButton();
            }
            
            function selectImage(index) {
                // Remove previous selection
                document.querySelectorAll('.image-box').forEach(box => {
                    box.classList.remove('selected');
                });
                
                // Add selection to clicked image
                document.querySelectorAll('.image-box')[index].classList.add('selected');
                
                selectedImageIndex = index;
                updateSubmitButton();
            }
            
            function updateSubmitButton() {
                const submitBtn = document.getElementById('submit-btn');
                submitBtn.disabled = selectedImageIndex === null;
            }
            
            async function submitResponse() {
                if (selectedImageIndex === null) return;
                
                try {
                    const response = await fetch(`${{API_ENDPOINT}}/submit-response`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            user_id: userId,
                            triplet_id: currentTriplet.triplet_id,
                            selected_image_index: selectedImageIndex,
                            response_time: Date.now() - currentTriplet.load_time,
                            session_id: currentTriplet.session_id
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        completedCount++;
                        updateProgress();
                        loadNextTriplet();
                    } else {
                        alert('Error submitting response. Please try again.');
                    }
                    
                } catch (error) {
                    console.error('Error submitting response:', error);
                    alert('Error submitting response. Please try again.');
                }
            }
            
            async function skipTriplet() {
                try {
                    await fetch(`${{API_ENDPOINT}}/skip-triplet`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            user_id: userId,
                            triplet_id: currentTriplet.triplet_id,
                            session_id: currentTriplet.session_id
                        })
                    });
                    
                    loadNextTriplet();
                    
                } catch (error) {
                    console.error('Error skipping triplet:', error);
                    loadNextTriplet(); // Continue anyway
                }
            }
            
            function updateProgress() {
                document.getElementById('completed-count').textContent = completedCount;
                document.getElementById('remaining-count').textContent = totalTriplets - completedCount;
                
                const progressPercent = (completedCount / totalTriplets) * 100;
                document.getElementById('progress-bar').style.width = `${{progressPercent}}%`;
            }
            
            function showCompletionMessage() {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('triplet-interface').style.display = 'none';
                document.getElementById('completion-message').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """.format(user_id=user_id)
    
    return html_template

# Lambda function for getting the next triplet
def get_triplet(event, context):
    """
    AWS Lambda function to get the next triplet for a user.
    """
    try:
        # Initialize AWS clients
        s3 = boto3.client('s3')
        dynamodb = boto3.resource('dynamodb')
        
        # Get environment variables
        image_bucket = os.environ['IMAGE_BUCKET']
        feedback_table = os.environ['FEEDBACK_TABLE']
        
        # Parse request
        query_params = event.get('queryStringParameters') or {}
        user_id = query_params.get('user_id', 'anonymous')
        
        # Get feedback table
        table = dynamodb.Table(feedback_table)
        
        # Check user's progress
        user_responses = table.query(
            IndexName='UserIndex',
            KeyConditionExpression='user_id = :user_id',
            ExpressionAttributeValues={':user_id': user_id}
        )
        
        completed_triplets = {item['triplet_id'] for item in user_responses['Items']}
        
        # Get available triplets from S3
        available_triplets = get_available_triplets(s3, image_bucket, completed_triplets)
        
        if not available_triplets:
            return {
                'statusCode': 200,
                'headers': cors_headers(),
                'body': json.dumps({
                    'completed': True,
                    'total_completed': len(completed_triplets)
                })
            }
        
        # Select next triplet
        next_triplet = available_triplets[0]
        
        # Generate signed URLs for images
        signed_urls = []
        for image_key in next_triplet['image_keys']:
            signed_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': image_bucket, 'Key': image_key},
                ExpiresIn=3600  # 1 hour
            )
            signed_urls.append(signed_url)
        
        triplet_data = {
            'triplet_id': next_triplet['triplet_id'],
            'images': signed_urls,
            'session_id': str(uuid.uuid4()),
            'load_time': datetime.utcnow().timestamp(),
            'total_triplets': len(available_triplets) + len(completed_triplets)
        }
        
        return {
            'statusCode': 200,
            'headers': cors_headers(),
            'body': json.dumps({
                'completed': False,
                'triplet': triplet_data,
                'total_triplets': triplet_data['total_triplets']
            })
        }
        
    except Exception as e:
        logging.error(f"Error getting triplet: {e}")
        return {
            'statusCode': 500,
            'headers': cors_headers(),
            'body': json.dumps({'error': str(e)})
        }

# Lambda function for submitting responses
def submit_response(event, context):
    """
    AWS Lambda function to submit user responses.
    """
    try:
        # Initialize AWS clients
        dynamodb = boto3.resource('dynamodb')
        
        # Get environment variables
        feedback_table = os.environ['FEEDBACK_TABLE']
        
        # Parse request
        body = json.loads(event['body'])
        
        # Validate required fields
        required_fields = ['user_id', 'triplet_id', 'selected_image_index']
        for field in required_fields:
            if field not in body:
                return {
                    'statusCode': 400,
                    'headers': cors_headers(),
                    'body': json.dumps({'error': f'Missing required field: {field}'})
                }
        
        # Get feedback table
        table = dynamodb.Table(feedback_table)
        
        # Create feedback record
        feedback_record = {
            'feedback_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': body['user_id'],
            'triplet_id': body['triplet_id'],
            'selected_image_index': body['selected_image_index'],
            'response_time_ms': body.get('response_time', 0),
            'session_id': body.get('session_id'),
            'user_agent': event.get('headers', {}).get('User-Agent', ''),
            'ip_address': event.get('requestContext', {}).get('identity', {}).get('sourceIp', ''),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store in DynamoDB
        table.put_item(Item=feedback_record)
        
        return {
            'statusCode': 200,
            'headers': cors_headers(),
            'body': json.dumps({'success': True, 'feedback_id': feedback_record['feedback_id']})
        }
        
    except Exception as e:
        logging.error(f"Error submitting response: {e}")
        return {
            'statusCode': 500,
            'headers': cors_headers(),
            'body': json.dumps({'error': str(e)})
        }

# Lambda function for skipping triplets
def skip_triplet(event, context):
    """
    AWS Lambda function to record skipped triplets.
    """
    try:
        # Initialize AWS clients
        dynamodb = boto3.resource('dynamodb')
        
        # Get environment variables
        feedback_table = os.environ['FEEDBACK_TABLE']
        
        # Parse request
        body = json.loads(event['body'])
        
        # Get feedback table
        table = dynamodb.Table(feedback_table)
        
        # Create skip record
        skip_record = {
            'feedback_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': body['user_id'],
            'triplet_id': body['triplet_id'],
            'action': 'skip',
            'session_id': body.get('session_id'),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store in DynamoDB
        table.put_item(Item=skip_record)
        
        return {
            'statusCode': 200,
            'headers': cors_headers(),
            'body': json.dumps({'success': True})
        }
        
    except Exception as e:
        logging.error(f"Error recording skip: {e}")
        return {
            'statusCode': 500,
            'headers': cors_headers(),
            'body': json.dumps({'error': str(e)})
        }

def get_available_triplets(s3, bucket: str, completed_triplets: set) -> List[Dict]:
    """Get available triplets from S3 that haven't been completed by the user."""
    triplets = []
    
    # List objects in the generated images bucket
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix='generated_images/'
    )
    
    # Group images into triplets
    # This is a simplified version - you may want to implement more sophisticated triplet creation
    image_keys = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.png') and 'image_' in obj['Key']:
            image_keys.append(obj['Key'])
    
    # Create triplets from available images
    for i in range(0, len(image_keys) - 2, 3):
        triplet_id = f"triplet_{i//3:06d}"
        
        if triplet_id not in completed_triplets:
            triplet = {
                'triplet_id': triplet_id,
                'image_keys': image_keys[i:i+3]
            }
            triplets.append(triplet)
    
    return triplets

def cors_headers():
    """Return CORS headers for API responses."""
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
    }
