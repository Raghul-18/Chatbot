# Car Repair Chatbot

A web application built with Streamlit that uses advanced AI models to provide actionable car repair instructions and visual aids. The chatbot leverages the Gemini API for generating repair guides and Stable Diffusion for creating images that illustrate each step.

## Features

- **Interactive Chatbot**: Describe your car problem, and get a detailed, actionable repair guide.
- **Image Generation**: Automatically generate images for each repair step using Stable Diffusion.
- **Actionable Steps**: Extract and display actionable steps for car repairs.
- **Search History**: Keep track of previous searches and display their results.
- **Customizable UI**: The app features a custom-styled sidebar and dynamic content presentation.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Raghul-18/Chatbot.git
    cd Chatbot
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**: Ensure you have the required API keys set up in your environment. For the Gemini API, set the `API_KEY` environment variable:
    ```bash
    export API_KEY=your_gemini_api_key
    ```

5. **Run the app**:
    ```bash
    streamlit run app.py
    ```

## Usage

- **Describe Your Car Problem**: Enter a description of the car issue in the input field.
- **Generate Prompt**: Click the "Generate Prompt" button to get a repair guide and images.
- **View Results**: See the generated instructions, actionable steps, and images for each step.
- **Access Search History**: Use the sidebar to view and select from previous searches.

## Contributing

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/your-feature
    ```
3. Make your changes.
4. Commit your changes:
    ```bash
    git commit -m "Add new feature"
    ```
5. Push to the branch:
    ```bash
    git push origin feature/your-feature
    ```
6. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Streamlit**: For creating an easy-to-use framework for building interactive web apps.
- **Hugging Face**: For providing powerful image generation models.
- **Google Gemini API**: For generating detailed and actionable repair guides.
