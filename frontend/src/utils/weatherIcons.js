import {WiDaySunny,WiRain,WiCloudy,WiThunderstorm} from 'react-icons/wi';

export const getWeatherIcon=(code) => {
    if(code>=200&&code<300) return <WiThunderstorm />;
    if(code>=300&&code<600) return <WiRain />;
    if(code>=600&&code<700) return <WiSnow />;
    if(code===800) return <WiDaySunny />;
    return <WiCloudy />;
};
