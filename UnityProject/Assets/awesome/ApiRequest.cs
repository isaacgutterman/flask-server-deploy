using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class ApiRequest : MonoBehaviour
{
    private string apiUrl = "http://127.0.0.1:5000"; // Change this to your deployed URL when ready
    public string localPath = "Assets/ResourcesMain/"; // Local path to save the files

    void Start()
    {
        StartCoroutine(UploadAndProcessCSV("original_csv_file_name.csv"));
    }

    IEnumerator UploadAndProcessCSV(string originalCsvFileName)
    {
        string originalCsvFilePath = Path.Combine(localPath, originalCsvFileName);
        yield return UploadCSVFile(originalCsvFilePath);

        yield return DownloadFile(apiUrl + "/data/MST_coord.csv", "MST_coord.csv");
        yield return DownloadFile(apiUrl + "/data/coordinates.csv", "coordinates.csv");

        AssignFiles(originalCsvFileName);
    }

    IEnumerator UploadCSVFile(string filePath)
    {
        WWWForm form = new WWWForm();
        byte[] fileData = File.ReadAllBytes(filePath);
        form.AddBinaryData("file", fileData, Path.GetFileName(filePath), "text/csv");

        UnityWebRequest uwr = UnityWebRequest.Post(apiUrl + "/upload", form);
        yield return uwr.SendWebRequest();

        if (uwr.result == UnityWebRequest.Result.ConnectionError || uwr.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.Log("Error: " + uwr.error);
        }
        else
        {
            Debug.Log("Received: " + uwr.downloadHandler.text);
        }
    }

    IEnumerator DownloadFile(string url, string fileName)
    {
        UnityWebRequest uwr = UnityWebRequest.Get(url);
        yield return uwr.SendWebRequest();

        if (uwr.result == UnityWebRequest.Result.ConnectionError || uwr.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.Log("Error: " + uwr.error);
        }
        else
        {
            File.WriteAllBytes(Path.Combine(localPath, fileName), uwr.downloadHandler.data);
            Debug.Log("File downloaded and saved: " + fileName);
        }
    }

    void AssignFiles(string originalCsvFileName)
    {
        // Paths to the downloaded files
        string coordinatesFilePath = Path.Combine(localPath, "coordinates.csv");
        string mstFilePath = Path.Combine(localPath, "MST_coord.csv");
        string originalCsvFilePath = Path.Combine(localPath, originalCsvFileName);

        // Load the files as TextAssets
        TextAsset coordinatesTextAsset = Resources.Load<TextAsset>("ResourcesMain/coordinates");
        TextAsset mstTextAsset = Resources.Load<TextAsset>("ResourcesMain/MST_coord");
        TextAsset originalCsvTextAsset = Resources.Load<TextAsset>("ResourcesMain/" + Path.GetFileNameWithoutExtension(originalCsvFilePath));

        // Find the relevant components in the scene
        MapGenerator mapGenerator = FindObjectOfType<MapGenerator>();
        CharacterSpawner characterSpawner = FindObjectOfType<CharacterSpawner>();
        HouseNames houseNames = FindObjectOfType<HouseNames>();

        // Assign the coordinates file
        mapGenerator.houseCsvFile = coordinatesTextAsset;
        characterSpawner.coordinatesCsvFile = coordinatesTextAsset;
        houseNames.HouseCords = coordinatesTextAsset;  // Correct field name for coordinates

        // Assign the MST file
        mapGenerator.mstCsvFile = mstTextAsset;

        // Assign the original CSV file
        characterSpawner.transcriptsCsvFile = originalCsvTextAsset;
        houseNames.Transcripts = originalCsvTextAsset;  // Correct field name for transcripts

        Debug.Log("Files assigned successfully");
    }
}
