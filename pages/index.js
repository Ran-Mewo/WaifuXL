import { getDataURIFromInput } from "../services/imageUtilities";
import { initializeONNX } from "../services/onnxBackend";
import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from "react-compare-slider";
import NavbarComponent from "../components/NavbarComponent";
import TitleComponent from "../components/TitleComponent";
import DownloadComponent from "../components/DownloadComponent";
import RunComponent from "../components/RunComponent";
import InputComponent from "../components/InputComponent";
import {useState, useEffect} from 'react';
export default function Example() {
  const [inputURI, setInputURI] = useState("https://i.imgur.com/Sf6sfPj.png");
  const [outputURI, setOutputURI] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inputModalOpen, setInputModalOpen] = useState(false);

  useEffect(async () => {
    await initializeONNX();
    //note: this is the input logic (given some from of URI)
    setInputURI(await getDataURIFromInput(inputURI));
  }, []);

  return (
    <>
      <div>
        <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
          <div className="flex-1 flex flex-col min-h-0 bg-gray-100">
            <div className="flex-1 flex flex-col overflow-y-auto">
              <div className="space-y-2 mx-8 grid grid-cols-1">
                <br/>
                <br/>
                <InputComponent
                  inputModalOpen={inputModalOpen}
                  setInputModalOpen={setInputModalOpen}
                  setInputURI={setInputURI}
                  setOutputURI={setOutputURI}
                />

                <RunComponent
                  setLoading={setLoading}
                  inputURI={inputURI}
                  setOutputURI={setOutputURI}
                />
                <br/>
                <br/>
                <hr />
                <br/>
                <br/>
                <div className="text-xl font-bold" style={{ textShadow: "white 0px 2px 4px" }}>Tags</div>
                <br/>

              </div>
            </div>
          </div>
        </div>
        <div className="md:pl-64 flex flex-col">
          <main className="flex-1">
            <div className="py-6">
              <div
                className="flex flex-col items-center min-h-screen"
                style={{
                  backgroundImage: `url("bg.png")`,
                  backgroundSize: "cover",
                }}
              >
                <NavbarComponent />
                <div className="flex absolute h-screen items-center justify-center">
                  {outputURI == null ? (
                    <img
                      src={inputURI}
                      className={"border-pink"}
                      style={{
                        height: 500,
                        borderWidth: "4px",
                        backgroundColor: "white",
                      }}
                    />
                  ) : (
                    <ReactCompareSlider
                      className={"border-pink"}
                      style={{
                        height: 500,
                        borderWidth: "4px",
                        backgroundColor: "white",
                      }}
                      itemOne={
                        <ReactCompareSliderImage
                          src={inputURI}
                          alt="Image one"
                        />
                      }
                      itemTwo={
                        <ReactCompareSliderImage
                          src={outputURI}
                          alt="Image two"
                        />
                      }
                    />
                  )}
                </div>
                <div className="absolute bottom-0">
                  {outputURI != null && (
                    <DownloadComponent
                      inputURI={inputURI}
                      outputURI={outputURI}
                    />
                  )}

                  <TitleComponent loading={loading} />
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
